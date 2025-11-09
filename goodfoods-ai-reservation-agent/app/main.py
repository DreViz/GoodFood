import streamlit as st
import json
import requests
import sseclient
import sys, os
from datetime import date, timedelta

# ---------- PATH SETUP ----------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- CONFIG ----------
st.set_page_config(
    page_title="GoodFoods Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/agent/chat/stream"   # streaming endpoint
RESET_MEMORY_URL = "http://127.0.0.1:8000/agent/memory/reset"  # new endpoint to clear context

# ---------- RESET MEMORY ON REFRESH ----------
if "memory_reset" not in st.session_state:
    try:
        requests.post(RESET_MEMORY_URL, timeout=5)
        st.session_state.memory_reset = True
        print(" Conversation memory reset (page refresh).")
    except Exception as e:
        print(f" Could not reset memory: {e}")

# ---------- GLOBAL JS ----------
st.markdown("""
<script>
setTimeout(() => {
  const doc = window.parent.document;
  const expandBtn = doc.querySelector('button[title="Expand sidebar"]');
  if (expandBtn) expandBtn.click();
}, 1000);
</script>
""", unsafe_allow_html=True)

# ---------- GLOBAL CSS ----------
st.markdown(
    """
    <style>
    :root{
      --gf-green: #4b6043;
      --gf-dark: #0f1112;
      --gf-accent: #D4A373;
      --gf-light: #f7f6f3;
      --gf-muted: #9a9a9a;
    }

    body { background-color: var(--gf-dark); }
    header[data-testid="stHeader"]{display:none;}
    .block-container { padding-top: 12px; padding-bottom: 20px; }

    section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #fbf9f6, #f7f4ef) !important;
      color: #222 !important;
      border-right: 1px solid rgba(0,0,0,0.06);
      min-width: 300px !important;
      max-width: 300px !important;
    }
    section[data-testid="stSidebar"] button[title="Collapse sidebar"],
    section[data-testid="stSidebar"] button[aria-label="Collapse sidebar"] {
        display: none !important;
    }

    .glass-card {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(8px) saturate(120%);
      -webkit-backdrop-filter: blur(8px) saturate(120%);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.6);
      margin-bottom: 18px;
      color: #eee;
    }
    .gf-title { font-size: 28px; font-weight: 700; color: var(--gf-accent); margin: 0 0 6px 0; }
    .gf-subtle { color: #c9c9c9; margin: 0 0 14px 0; }

    .chat-bubble-user {
      background: #f7f6f3; color: #111;
      border-radius: 12px; padding: 10px 13px; margin: 8px 0;
      display: inline-block; max-width: 92%;
      border: 1px solid rgba(0,0,0,0.06);
    }
    .chat-bubble-assistant {
      background: rgba(255,255,255,0.03); color: #eaeaea;
      border-radius: 12px; padding: 10px 13px; margin: 8px 0;
      display: inline-block; max-width: 92%;
      border: 1px solid rgba(255,255,255,0.04);
    }

    div[data-testid="stChatInput"] {
      background: #202223 !important;
      border-radius: 28px !important;
      padding: 10px 16px !important;
      border: 1px solid rgba(255,255,255,0.04) !important;
    }

    section[data-testid="stSidebar"] * { color: #222 !important; }
    section[data-testid="stSidebar"] button {
      background-color: #4b6043 !important; color: #fff !important;
      border-radius: 8px !important; border: none !important;
    }
    section[data-testid="stSidebar"] button:hover {
      background-color: #3b4f35 !important;
    }

    @media (max-width: 900px){
      .glass-card { padding: 12px; border-radius: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        """
        <div style="margin-top:6px; margin-bottom:8px;">
            <h3 style="margin:0;color:#222;">🥗 GoodFoods</h3>
            <p style="margin:4px 0 0 0;color:#555;font-size:0.92rem;">
            Your personal dining concierge — save preferences for smarter suggestions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    name = st.text_input("Your Name", key="user_name")
    email = st.text_input("Email", key="user_email")

    cuisine = st.selectbox(
        "Cuisine",
        ["Any", "South Indian", "North Indian", "Pan-Asian", "Italian", "Mediterranean", "BBQ", "Continental"],
        key="pref_cuisine",
    )

    price = st.select_slider(
        "Max price per person (₹)",
        options=[300, 400, 500, 600, 700, 800, 900, 1000, 1200],
        value=800,
        key="pref_price",
    )

    vibe = st.multiselect(
        "Vibe / Tags",
        ["family-friendly", "date-night", "business-lunch", "outdoor", "live-music", "healthy", "premium"],
        key="pref_vibe",
    )

    avoid_music = st.checkbox("Prefer quieter places (avoid loud music)", value=False, key="pref_quiet")

    st.markdown("**Select Preferred Date**")
    today = date.today()
    next_7_days = [today + timedelta(days=i) for i in range(7)]
    date_labels = [d.strftime("%a %d %b") for d in next_7_days]
    selected_label = st.radio("Choose date", options=date_labels, horizontal=True, key="pref_date")
    selected_date = next_7_days[date_labels.index(selected_label)].strftime("%Y-%m-%d")

    seating_pref = st.selectbox("Seating Preference", ["Any", "outdoor", "indoor", "window-side"], key="pref_seating")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(" Save My Preferences"):
        if not name or not email:
            st.error("Please enter both name and email before saving preferences.")
        else:
            payload = {
                "name": name.strip() or None,
                "email": email.strip() or None,
                "allergies": None,
                "preferred_cuisines": [cuisine] if cuisine != "Any" else None,
                "avoid_music": True if avoid_music else None,
                "seating_preference": seating_pref if seating_pref != "Any" else None,
                "date": selected_date if selected_date else None,
                "vibe_tags": ", ".join(vibe) if vibe else None,
                "max_price": price if price else None,
            }
            try:
                res = requests.post("http://127.0.0.1:8000/customers/profile", json=payload, timeout=8)
                if res.status_code == 200:
                    st.success(" Preferences saved successfully!")
                else:
                    st.error(f" Couldn’t save preferences: {res.status_code} {res.text}")
            except Exception as e:
                st.error(f"Error saving preferences: {e}")

    st.caption("💡 Tip: Try ‘Find an outdoor table for 4 in Indiranagar under ₹800’")

# ---------- HEADER CARD ----------
st.markdown(
    """
    <div class="glass-card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap;">
        <div style="flex:1;min-width:280px;">
          <div class="gf-title">GoodFoods Reservation Assistant</div>
          <div class="gf-subtle">Discover, chat, and instantly reserve the perfect dining spot — powered by AI 🍽️</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    bubble_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-assistant"
    st.markdown(f'<div class="{bubble_class}">{msg["content"]}</div>', unsafe_allow_html=True)


def build_context_from_history(messages, limit=10):
    recent = messages[-limit:]
    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted).strip()


# ---------- CHAT INPUT + STREAMING ----------
user_text = st.chat_input("Ask, book, or explore restaurants…")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.markdown(f'<div class="chat-bubble-user">{user_text}</div>', unsafe_allow_html=True)

    history_text = build_context_from_history(st.session_state.messages, limit=8)
    placeholder = st.empty()
    full_response = ""

    # 🧠 Update backend memory before chatting
    try:
        mem_res = requests.post("http://127.0.0.1:8000/agent/memory/update", json={"text": user_text}, timeout=6)
        if mem_res.status_code != 200:
            st.warning("⚠️ Memory update failed, continuing without it.")
    except Exception as e:
        st.warning(f"⚠️ Could not sync conversation memory: {e}")

    with st.chat_message("assistant"):
        st.markdown('<div class="chat-bubble-assistant">', unsafe_allow_html=True)
        try:
            payload = {"query": user_text, "context": history_text}
            with requests.post(API_URL, json=payload, stream=True, timeout=60) as resp:
                client = sseclient.SSEClient(resp)
                for event in client.events():
                    if not event.data:
                        continue
                    if event.data == "[DONE]":
                        break
                    try:
                        data = json.loads(event.data)
                        if "message" in data:
                            full_response += data["message"]
                            placeholder.markdown(
                                f'<div class="chat-bubble-assistant">{full_response}▌</div>',
                                unsafe_allow_html=True,
                            )
                        elif "tool_output" in data:
                            placeholder.markdown(
                                f'<div class="chat-bubble-assistant">{full_response}<br><pre style="background:#0f1112;color:#dcdcdc;padding:10px;border-radius:8px;">{json.dumps(data["tool_output"], indent=2)}</pre></div>',
                                unsafe_allow_html=True,
                            )
                    except Exception:
                        full_response += event.data
                        placeholder.markdown(
                            f'<div class="chat-bubble-assistant">{full_response}▌</div>',
                            unsafe_allow_html=True,
                        )
            placeholder.markdown(
                f'<div class="chat-bubble-assistant">{full_response}</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            placeholder.markdown(
                f'<div class="chat-bubble-assistant">Sorry — streaming failed: {e}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
