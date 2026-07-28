"use client";

import { useState } from "react";
import { ChevronDown, Wrench } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface AgentTrace {
  type: string;
  phase?: string;
  tool_output?: {
    action?: string;
    args?: unknown;
    result?: unknown;
  } | null;
}

// Collapsible panel rendering the planner action/args/tool-result carried on an
// assistant message's AI SDK annotations.
export function ToolTrace({ annotations }: { annotations?: unknown[] }) {
  const [open, setOpen] = useState(false);

  const trace = (annotations ?? []).find(
    (a): a is AgentTrace =>
      !!a && typeof a === "object" && (a as AgentTrace).type === "agent_trace",
  );
  if (!trace) return null;

  const action = trace.tool_output?.action;
  const payload = trace.tool_output ?? { phase: trace.phase };

  return (
    <div className="mt-2 overflow-hidden rounded-lg border border-border/60 bg-muted/40 text-xs">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-muted-foreground transition-colors hover:text-foreground"
      >
        <Wrench className="size-3.5" />
        <span className="font-medium">Agent trace</span>
        {trace.phase ? (
          <Badge variant="secondary" className="h-5 px-1.5 text-[10px] font-normal">
            {trace.phase}
          </Badge>
        ) : null}
        {action ? (
          <Badge className="h-5 px-1.5 text-[10px] font-normal">{action}</Badge>
        ) : null}
        <ChevronDown
          className={`ml-auto size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open ? (
        <pre className="overflow-x-auto border-t border-border/60 p-3 text-[11px] leading-relaxed text-muted-foreground">
          {JSON.stringify(payload, null, 2)}
        </pre>
      ) : null}
    </div>
  );
}
