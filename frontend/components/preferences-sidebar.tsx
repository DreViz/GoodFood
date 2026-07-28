"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Save, SlidersHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { saveProfile } from "@/lib/api";

const CUISINES = [
  "Italian",
  "North Indian",
  "South Indian",
  "Chinese",
  "Continental",
  "Pan-Asian",
  "Mediterranean",
  "BBQ",
];

const SEATING = ["Indoor", "Outdoor", "Bar", "Private"];

export function PreferencesSidebar() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [allergies, setAllergies] = useState("");
  const [cuisines, setCuisines] = useState<string[]>([]);
  const [seating, setSeating] = useState<string | null>(null);
  const [avoidMusic, setAvoidMusic] = useState(false);
  const [saving, setSaving] = useState(false);

  function toggleCuisine(c: string) {
    setCuisines((prev) =>
      prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c],
    );
  }

  async function handleSave() {
    if (!name.trim() || !email.trim()) {
      toast.error("Name and email are required.");
      return;
    }
    setSaving(true);
    try {
      await saveProfile({
        name: name.trim(),
        email: email.trim(),
        allergies: allergies
          .split(",")
          .map((a) => a.trim())
          .filter(Boolean),
        preferred_cuisines: cuisines,
        avoid_music: avoidMusic,
        seating_preference: seating,
      });
      toast.success("Preferences saved.");
    } catch {
      toast.error("Couldn't save preferences.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex flex-col gap-5 p-5">
      <div className="flex items-center gap-2">
        <SlidersHorizontal className="size-4 text-primary" />
        <h2 className="text-sm font-semibold">Your preferences</h2>
      </div>

      <Field label="Name">
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Jane Doe" />
      </Field>
      <Field label="Email">
        <Input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="jane@example.com"
        />
      </Field>

      <Separator />

      <Field label="Preferred cuisines">
        <div className="flex flex-wrap gap-1.5">
          {CUISINES.map((c) => {
            const active = cuisines.includes(c);
            return (
              <button
                key={c}
                type="button"
                onClick={() => toggleCuisine(c)}
                className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                  active
                    ? "border-primary bg-primary/15 text-primary"
                    : "border-border text-muted-foreground hover:text-foreground"
                }`}
              >
                {c}
              </button>
            );
          })}
        </div>
      </Field>

      <Field label="Seating preference">
        <div className="flex flex-wrap gap-1.5">
          {SEATING.map((s) => {
            const active = seating === s;
            return (
              <button
                key={s}
                type="button"
                onClick={() => setSeating(active ? null : s)}
                className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                  active
                    ? "border-primary bg-primary/15 text-primary"
                    : "border-border text-muted-foreground hover:text-foreground"
                }`}
              >
                {s}
              </button>
            );
          })}
        </div>
      </Field>

      <Field label="Allergies">
        <Input
          value={allergies}
          onChange={(e) => setAllergies(e.target.value)}
          placeholder="peanuts, shellfish…"
        />
        <p className="mt-1 text-[11px] text-muted-foreground">Comma-separated.</p>
      </Field>

      <label className="flex cursor-pointer items-center justify-between text-sm">
        <span>Prefer a quiet table</span>
        <button
          type="button"
          role="switch"
          aria-checked={avoidMusic}
          onClick={() => setAvoidMusic((v) => !v)}
          className={`relative h-5 w-9 rounded-full transition-colors ${
            avoidMusic ? "bg-primary" : "bg-muted"
          }`}
        >
          <span
            className={`absolute top-0.5 size-4 rounded-full bg-background transition-transform ${
              avoidMusic ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>
      </label>

      <Button onClick={handleSave} disabled={saving} className="mt-1 gap-1.5">
        <Save className="size-4" />
        {saving ? "Saving…" : "Save preferences"}
      </Button>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-muted-foreground">{label}</label>
      {children}
    </div>
  );
}
