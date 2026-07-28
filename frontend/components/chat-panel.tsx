"use client";

import { useEffect, useRef } from "react";
import { useChat, type Message } from "@ai-sdk/react";
import { toast } from "sonner";
import { Bot, RotateCcw, Send, User } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ToolTrace } from "@/components/tool-trace";
import { API_URL, resetMemory } from "@/lib/api";

const SUGGESTIONS = [
  "Any Italian places in the East zone?",
  "Book a table for 2 at GoodFoods Bistro",
  "What seating options does it have?",
];

export function ChatPanel() {
  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    status,
    setMessages,
    append,
  } = useChat({
    api: `${API_URL}/agent/chat/stream`,
    streamProtocol: "data",
    // Backend expects ChatRequest {message, context}; send the latest user turn.
    experimental_prepareRequestBody: ({ messages }) => {
      const last = messages[messages.length - 1];
      return { message: last?.content ?? "", context: "" };
    },
    onError: () =>
      toast.error("Couldn't reach the assistant. Is the backend running on :8000?"),
  });

  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, status]);

  const busy = status === "submitted" || status === "streaming";
  const lastIsUser = messages[messages.length - 1]?.role === "user";

  async function handleReset() {
    try {
      await resetMemory();
      setMessages([]);
      toast.success("Started a new conversation.");
    } catch {
      toast.error("Failed to reset the conversation.");
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="flex size-8 items-center justify-center rounded-full bg-primary/15 text-primary">
            <Bot className="size-4" />
          </span>
          <div>
            <p className="text-sm font-semibold leading-none">GoodFoods Concierge</p>
            <p className="text-xs text-muted-foreground">Reservations by chat</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleReset}
          className="gap-1.5 text-muted-foreground"
        >
          <RotateCcw className="size-3.5" />
          New chat
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="mx-auto flex max-w-2xl flex-col gap-4 px-4 py-6">
          {messages.length === 0 ? (
            <EmptyState onPick={(t) => append({ role: "user", content: t })} />
          ) : (
            messages.map((m) => <MessageBubble key={m.id} message={m} />)
          )}
          {busy && lastIsUser ? <TypingIndicator /> : null}
          <div ref={endRef} />
        </div>
      </ScrollArea>

      <form onSubmit={handleSubmit} className="border-t border-border p-3">
        <div className="mx-auto flex max-w-2xl items-center gap-2">
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask, book, or explore restaurants…"
            disabled={busy}
            className="flex-1"
            autoFocus
          />
          <Button type="submit" size="icon" disabled={busy || !input.trim()}>
            <Send className="size-4" />
          </Button>
        </div>
      </form>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <Avatar className="size-8 shrink-0">
        <AvatarFallback
          className={
            isUser ? "bg-accent text-accent-foreground" : "bg-primary/15 text-primary"
          }
        >
          {isUser ? <User className="size-4" /> : <Bot className="size-4" />}
        </AvatarFallback>
      </Avatar>
      <div className={`flex min-w-0 max-w-[85%] flex-col ${isUser ? "items-end" : ""}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? "rounded-tr-sm bg-accent text-accent-foreground"
              : "rounded-tl-sm border border-border bg-card text-card-foreground"
          }`}
        >
          {message.content || <span className="text-muted-foreground">…</span>}
        </div>
        {!isUser ? <ToolTrace annotations={message.annotations} /> : null}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <Avatar className="size-8 shrink-0">
        <AvatarFallback className="bg-primary/15 text-primary">
          <Bot className="size-4" />
        </AvatarFallback>
      </Avatar>
      <div className="flex items-center gap-1 rounded-2xl rounded-tl-sm border border-border bg-card px-4 py-3.5">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="size-1.5 animate-bounce rounded-full bg-muted-foreground/70"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="flex flex-col items-center gap-5 py-16 text-center">
      <span className="flex size-14 items-center justify-center rounded-2xl bg-primary/15 text-primary">
        <Bot className="size-7" />
      </span>
      <div className="space-y-1">
        <h2 className="text-lg font-semibold">How can I help you dine today?</h2>
        <p className="max-w-sm text-sm text-muted-foreground">
          Search restaurants, check availability, and book a table — all in plain
          language.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onPick(s)}
            className="rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
