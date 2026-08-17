// Centralised API helper for the Next.js frontend.
// All fetches go to the FastAPI backend; the base URL is configurable via
// NEXT_PUBLIC_API_URL so the same build works in dev, preview, and prod.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Preferred typing shared with the backend Customer model.
export interface CustomerProfile {
  name: string;
  email: string;
  allergies?: string[] | null;
  preferred_cuisines?: string[] | null;
  avoid_music?: boolean | null;
  seating_preference?: string | null;
}

// Clear server-side conversation memory. Called when the user starts a new chat.
export async function resetMemory(signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_URL}/agent/memory/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
  });
  if (!res.ok) {
    throw new Error(`memory reset failed: ${res.status}`);
  }
}

// Save or update the customer profile (sidebar form).
export async function saveProfile(
  profile: CustomerProfile,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/customers/profile`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
    signal,
  });
  if (!res.ok) {
    throw new Error(`profile save failed: ${res.status}`);
  }
}
