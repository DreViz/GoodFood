import { UtensilsCrossed } from "lucide-react";
import { ChatPanel } from "@/components/chat-panel";
import { PreferencesSidebar } from "@/components/preferences-sidebar";

export default function Home() {
  return (
    <div className="flex h-dvh flex-col bg-background">
      <header className="flex items-center gap-2.5 border-b border-border px-5 py-3">
        <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <UtensilsCrossed className="size-4" />
        </span>
        <div>
          <h1 className="text-sm font-semibold leading-none">GoodFoods</h1>
          <p className="text-xs text-muted-foreground">AI Reservation Concierge</p>
        </div>
        <span className="ml-auto rounded-full border border-border px-2.5 py-1 text-[11px] text-muted-foreground">
          qwen3 · local LLM
        </span>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[1fr_20rem]">
        <main className="min-h-0 overflow-hidden">
          <ChatPanel />
        </main>
        <aside className="hidden min-h-0 overflow-y-auto border-l border-border lg:block">
          <PreferencesSidebar />
        </aside>
      </div>
    </div>
  );
}
