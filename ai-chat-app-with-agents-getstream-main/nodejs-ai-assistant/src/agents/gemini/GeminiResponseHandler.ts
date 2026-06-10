import type { Channel, Event, MessageResponse, StreamChat } from "stream-chat";

export class GeminiResponseHandler {
  private message_text = "";
  private is_done = false;
  private last_update_time = 0;

  constructor(
    private readonly chat: any,
    private readonly responseStream: any,
    private readonly chatClient: StreamChat,
    private readonly channel: Channel,
    private readonly message: MessageResponse,
    private readonly onDispose: () => void
  ) {
    this.chatClient.on("ai_indicator.stop", this.handleStopGenerating);
  }

  run = async () => {
    const { cid, id: message_id } = this.message;
    let isCompleted = false;
    let hasFunctionCalls = false;
    let functionCallsToExecute: any[] = [];
    let responseStream = this.responseStream;
    let hasGenerated = false;

    try {
      while (!isCompleted) {
        for await (const chunk of responseStream) {
          // Check if the user requested to stop generating
          if (this.is_done) {
            break;
          }

          // Check for function calls
          if (chunk.functionCalls && chunk.functionCalls.length > 0) {
            hasFunctionCalls = true;
            functionCallsToExecute.push(...chunk.functionCalls);

            await this.channel.sendEvent({
              type: "ai_indicator.update",
              ai_state: "AI_STATE_EXTERNAL_SOURCES",
              cid: cid,
              message_id: message_id,
            });

            break; // Break the chunk loop to process function calls
          }

          // Handle text chunks
          if (chunk.text) {
            if (!hasGenerated) {
              hasGenerated = true;
              await this.channel.sendEvent({
                type: "ai_indicator.update",
                ai_state: "AI_STATE_GENERATING",
                cid: cid,
                message_id: message_id,
              });
            }

            this.message_text += chunk.text;
            const now = Date.now();
            if (now - this.last_update_time > 1000) {
              await this.chatClient.partialUpdateMessage(message_id, {
                set: { text: this.message_text },
              });
              this.last_update_time = now;
            }
          }
        }

        if (this.is_done) {
          break;
        }

        if (hasFunctionCalls && functionCallsToExecute.length > 0) {
          const parts: any[] = [];
          for (const call of functionCallsToExecute) {
            if (call.name === "web_search") {
              try {
                const query = call.args?.query;
                const searchResult = await this.performWebSearch(query);
                parts.push({
                  functionResponse: {
                    name: call.name,
                    response: { result: searchResult },
                    ...(call.id ? { id: call.id } : {}),
                  },
                });
              } catch (e) {
                console.error("Error executing web search function call", e);
                parts.push({
                  functionResponse: {
                    name: call.name,
                    response: { error: "failed to call tool" },
                    ...(call.id ? { id: call.id } : {}),
                  },
                });
              }
            }
          }

          // Reset triggers
          hasFunctionCalls = false;
          functionCallsToExecute = [];

          // Submit function response back to the chat and get a new stream
          responseStream = await this.chat.sendMessageStream({
            message: parts,
          });

          continue; // Continue the loop to stream the model's response to the function result
        }

        // If we finished the stream without finding any more function calls, we are done
        isCompleted = true;
      }

      if (!this.is_done) {
        // Final message update with complete text
        await this.chatClient.partialUpdateMessage(message_id, {
          set: { text: this.message_text },
        });
        await this.channel.sendEvent({
          type: "ai_indicator.clear",
          cid: cid,
          message_id: message_id,
        });
      }
    } catch (error) {
      console.error("An error occurred during the Gemini run:", error);
      await this.handleError(error as Error);
    } finally {
      await this.dispose();
    }
  };

  dispose = async () => {
    if (this.is_done) {
      return;
    }
    this.is_done = true;
    this.chatClient.off("ai_indicator.stop", this.handleStopGenerating);
    this.onDispose();
  };

  private handleStopGenerating = async (event: Event) => {
    if (this.is_done || event.message_id !== this.message.id) {
      return;
    }

    console.log("Stop generating for message", this.message.id);
    await this.channel.sendEvent({
      type: "ai_indicator.clear",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.dispose();
  };

  private handleError = async (error: Error) => {
    if (this.is_done) {
      return;
    }
    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_ERROR",
      cid: this.message.cid,
      message_id: this.message.id,
    });
    await this.chatClient.partialUpdateMessage(this.message.id, {
      set: {
        text: error.message ?? "Error generating the message",
        message: error.toString(),
      },
    });
    await this.dispose();
  };

  private performWebSearch = async (query: string): Promise<string> => {
    const TAVILY_API_KEY = process.env.TAVILY_API_KEY;

    if (!TAVILY_API_KEY) {
      return JSON.stringify({
        error: "Web search is not available. API key not configured.",
      });
    }

    console.log(`Performing web search for: "${query}"`);

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TAVILY_API_KEY}`,
        },
        body: JSON.stringify({
          query: query,
          search_depth: "advanced",
          max_results: 5,
          include_answer: true,
          include_raw_content: false,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Tavily search failed for query "${query}":`, errorText);
        return JSON.stringify({
          error: `Search failed with status: ${response.status}`,
          details: errorText,
        });
      }

      const data = await response.json();
      console.log(`Tavily search successful for query "${query}"`);

      return JSON.stringify(data);
    } catch (error) {
      console.error(
        `An exception occurred during web search for "${query}":`,
        error
      );
      return JSON.stringify({
        error: "An exception occurred during the search.",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  };
}
