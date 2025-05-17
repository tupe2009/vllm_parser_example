# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Sequence
from typing import Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, DeltaFunctionCall, DeltaMessage,
    DeltaToolCall, ExtractedToolCallInformation, FunctionCall, ToolCall
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (
    consume_space, find_common_prefix, is_complete_json, partial_json_loads,
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)

@ToolParserManager.register_module("command_a")
class CommandAToolParser(ToolParser):
    """
    Tool call parser for Command-A models using <|START_ACTION|> markers.
    Handles responses formatted as:
    <|START_THINKING|>...<|END_THINKING|> <|START_ACTION|>[...]<|END_ACTION|>
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.thinking_start_token = "<|START_THINKING|>"
        self.thinking_end_token = "<|END_THINKING|>"
        print("initializing tool parser!!!!!!!!!!!!!!!!!!")
        self.action_start_token = "<|START_ACTION|>"
        self.action_end_token = "<|END_ACTION|>"
        self.action_regex = re.compile(
            rf"{re.escape(self.action_start_token)}(.*?){re.escape(self.action_end_token)}",
            re.DOTALL
        )
        self.in_thinking_mode=False
        self.done_thinking= False

        self.in_action_mode=False
        self.done_action= False
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self.prev_tool_call_arr = []
        self.current_tool_name_sent = False

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        print("---------------this line is by ofek ---------------------\n\n\n")
        print(model_output)
        if self.action_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

        try:
            # Extract content between action markers
            match = self.action_regex.search(model_output)
            if not match:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output)

            raw_actions = match.group(1).strip()
            tool_calls = []
            
            # Parse JSON array of tool calls
            action_list = json.loads(raw_actions)
            for action in action_list:
                tool_calls.append(ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=action["name"],
                        arguments=json.dumps(action["arguments"])
                    )
                ))

            # Extract non-tool content
            content = model_output.split(self.action_start_token)[0].strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None
            )

        except Exception as e:
            logger.error(f"Error parsing command-a tool calls: {e}")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
#        print("in streaming tool parsing------------------------------")
        if not self.in_thinking_mode and  not self.done_thinking and self.thinking_start_token in current_text:
            self.in_thinking_mode = True
            #If there is anything after the tag, return it as content
            current_text_delta = current_text.replace(self.thinking_start_token,"")
            if current_text_delta:
                return DeltaMessage(content=current_text_delta)
        if self.in_thinking_mode and not self.done_thinking and self.thinking_end_token in current_text:
            print("now out of thinking mode!!!!!!!!")
            self.done_thinking = True
            #if there is anything before the tag that was not streamed, stream it as content
            #this may cause issues if the token does not come in full in a single token but split into different one.
            #Ill fix this later
            if self.thinking_end_token in delta_text:
                delta_content_left = delta_text.replace(self.thinking_end_token,"")
                if delta_content_left:
                    return DeltaMessage(content=delta_content_left)
        if not self.in_action_mode and not self.done_action and self.action_start_token in current_text:
            print("now in action mode!!!!")
            self.in_action_mode = True
        if self.in_thinking_mode and not self.done_thinking:
            return DeltaMessage(content=delta_text)
        elif self.in_action_mode:

            # bit mask flags for partial JSON parsing. If the name hasn't been
            # sent yet, don't allow sending
            # an incomplete string since OpenAI only ever (as far as I have
            # seen) allows sending the entire tool/ function name at once.
            flags = Allow.ALL if self.current_tool_name_sent \
                else Allow.ALL & ~Allow.STR

            # Extract JSON content between action markers
            action_content = current_text.split(self.action_start_token)[-1].split(self.action_end_token)[0]                
            print(action_content)
            parsable_arr = action_content
            if not action_content:
                return None
            # tool calls are generated in an array, so do partial JSON
            # parsing on the entire array
            try:
                tool_call_arr: list[dict] = partial_json_parser.loads(
                    parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug('not enough tokens to parse into JSON yet')
                return None
            except json.decoder.JSONDecodeError:
                logger.debug('Json Decode error. probably means the partial_json_parser does not have enough tokens yet')
                return None

            # select as the current tool call the one we're on the state at
            for item in tool_call_arr:
                if "tool_name" in item:
                    item["name"] = item.pop("tool_name")
                if "parameters" in item:
                    item["arguments"] = item.pop("parameters")


            current_tool_call: dict = tool_call_arr[self.current_tool_id] \
                if len(tool_call_arr) > 0 else {}

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return None

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (len(tool_call_arr) > 0
                and len(tool_call_arr) > self.current_tool_id + 1):
                print(f"found a new tool, value of current_tool_id is: {self.current_tool_id} ")
                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    diff: Union[str, None] = current_tool_call.get("arguments")

                    if diff:
                        diff = json.dumps(diff, ensure_ascii=False).replace(
                            self.streamed_args_for_tool[self.current_tool_id],
                            "")
                        delta = DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id,
                                        function=DeltaFunctionCall(
                                            arguments=diff).model_dump(
                                                exclude_none=True))
                        ])
                        self.streamed_args_for_tool[
                            self.current_tool_id] += diff
                    else:
                        delta = None
                else:
                    delta = None
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("starting on new tool %d", self.current_tool_id)
                return delta

            # case: update an existing tool - this is handled below

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            print(f"value of self.current_tool_name_sent: {str(self.current_tool_name_sent)}")
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                print(f"value of function name: {str(function_name)}")

                if function_name:
                    new_tool_id = str(random_uuid())

                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                    type="function",
                                    id=new_tool_id,
                                    function=DeltaFunctionCall(
                                        name=function_name).model_dump(
                                            exclude_none=True))
                    ])
                    self.current_tool_name_sent = True
                else:
                    delta = None

            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                print("now we know we're on the same tool call and we're streaming")
                prev_arguments = self.prev_tool_call_arr[
                    self.current_tool_id].get("arguments")
                cur_arguments = current_tool_call.get("arguments")
                new_text = delta_text
                if ('"}' in new_text):
                    new_text = new_text[:new_text.rindex('"}')]

                if not cur_arguments and not prev_arguments:

                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error(
                        "INVARIANT - impossible to have arguments reset "
                        "mid-arguments")
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments,
                                                    ensure_ascii=False)[:-2]
                    logger.debug("finding %s in %s", new_text,
                                cur_arguments_json)

                    if (new_text not in cur_arguments_json):
                        return None
                    print(f"current arg json is {cur_arguments_json}")
                    arguments_delta = cur_arguments_json[:cur_arguments_json.
                                                        rindex(new_text) +
                                                        len(new_text)]
                    logger.debug("First tokens in arguments received: %s",
                                arguments_delta)
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=arguments_delta).
                                    model_dump(exclude_none=True))
                    ])
                    self.streamed_args_for_tool[
                        self.current_tool_id] += arguments_delta

                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments,
                                            ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments,
                                                ensure_ascii=False)
                    logger.debug("Searching for diff between \n%s\n%s",
                                cur_args_json, prev_args_json)

                    argument_diff = extract_intermediate_diff(
                        cur_args_json, prev_args_json)
                    logger.debug("got arguments diff: %s", argument_diff)
                    print(f"got arguments    diff: {argument_diff}")
                    delta = DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                    function=DeltaFunctionCall(
                                        arguments=argument_diff).model_dump(
                                            exclude_none=True))
                    ])
                    self.streamed_args_for_tool[
                        self.current_tool_id] += argument_diff
                else:
                    # try parsing it with regular JSON - if it works we're
                    # at the end, and we need to send the difference between
                    # tokens streamed so far and the valid JSON
                    delta = None

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            self.prev_tool_call_arr = tool_call_arr
            if delta is not None and self.in_action_mode:
                print(f"returned args are: {delta.tool_calls[0].function.arguments}")
            return delta

