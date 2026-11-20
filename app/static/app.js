const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const retrieverInput = document.querySelector("#retriever");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const charCount = document.querySelector("#char-count");

function resizeInput() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
}

function addMessage(role, text) {
  const message = document.createElement("article");
  message.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "Y" : "✳";

  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "GPU DevAssist";
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;
  body.append(label, content);
  message.append(avatar, body);
  messages.append(message);
  return { message, content };
}

// Display names for the indexed PDFs; unknown files fall back to the raw filename
const DOCUMENT_NAMES = {
  "oneapi_optimization-guide-gpu_2025.0-771772-834474.pdf": "oneAPI GPU Optimization Guide 2025.0",
  "oneapi_programming-guide_2025.1-771723-848694.pdf": "oneAPI Programming Guide 2025.1",
};

function addSources(container, sources) {
  if (!sources.length) return;

  const card = document.createElement("div");
  card.className = "sources-card";
  const heading = document.createElement("div");
  heading.className = "sources-heading";
  heading.innerHTML = '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M4 2.5h5l3 3v8H4v-11Z"/><path d="M9 2.5v3h3M6 8h4M6 10.5h4"/></svg> Sources';
  const list = document.createElement("div");
  list.className = "source-list";
  for (const source of sources) {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    const number = document.createElement("span");
    number.className = "source-number";
    number.textContent = `[${source.id}]`;
    const citation = document.createElement("span");
    citation.textContent = `${DOCUMENT_NAMES[source.source] ?? source.source}, p. ${source.page}`;
    chip.append(number, citation);
    list.append(chip);
  }
  card.append(heading, list);
  container.append(card);
}

function showError(message) {
  const error = document.createElement("p");
  error.className = "message-content error-text";
  error.setAttribute("role", "alert");
  error.textContent = message;
  return error;
}

async function ask(question) {
  const userMessage = question.trim();
  if (userMessage.length < 3 || userMessage.length > 500) return;

  welcome.hidden = true;
  addMessage("user", userMessage);
  questionInput.value = "";
  questionInput.style.height = "auto";
  charCount.textContent = "0 / 500";
  sendButton.disabled = true;

  const { content } = addMessage("assistant", "");
  const pending = document.createElement("div");
  pending.className = "pending";
  pending.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Searching documentation...</span>';
  content.append(pending);
  content.parentElement.parentElement.setAttribute("aria-busy", "true");
  content.scrollIntoView({ block: "nearest", behavior: "smooth" });

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: userMessage, retriever: retrieverInput.value }),
    });
    const result = await response.json();
    if (!response.ok) {
      const detail = typeof result.detail === "string"
        ? result.detail
        : "Please check your question and try again.";
      throw new Error(detail);
    }

    pending.remove();
    const answer = document.createElement("div");
    answer.className = "answer-text";
    answer.textContent = result.answer;
    content.append(answer);
    addSources(content, result.sources || []);

    const meta = document.createElement("div");
    meta.className = "answer-meta";
    const elapsed = document.createElement("span");
    elapsed.textContent = `${result.latency_ms} ms`;
    const method = document.createElement("span");
    method.textContent = `${result.retriever} search`;
    meta.append(elapsed, method);
    content.append(meta);
  } catch (error) {
    pending.remove();
    content.append(showError(error instanceof Error ? error.message : "Something went wrong. Please try again."));
  } finally {
    content.parentElement.parentElement.removeAttribute("aria-busy");
    sendButton.disabled = false;
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(questionInput.value);
});

questionInput.addEventListener("input", () => {
  charCount.textContent = `${questionInput.value.length} / 500`;
  resizeInput();
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => ask(button.dataset.question));
});

document.querySelector("#new-chat").addEventListener("click", () => {
  messages.replaceChildren();
  welcome.hidden = false;
  questionInput.value = "";
  questionInput.style.height = "auto";
  charCount.textContent = "0 / 500";
  questionInput.focus();
});
