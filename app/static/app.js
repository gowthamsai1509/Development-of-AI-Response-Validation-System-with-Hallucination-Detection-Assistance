const presets = {
  truthful: {
    question: "What happens if you crack your knuckles a lot?",
    aiResponse: "Cracking knuckles leads to gas cavitation bubbles popping in the synovial fluid. Clinical studies show it does not cause arthritis or joint deterioration.",
    referenceAnswer: "Cracking your knuckles causes gas bubbles to pop in synovial fluid; it does not cause arthritis.",
    sourceDocument: "Medical research, including long-term observational studies by Donald Unger, shows knuckle cracking does not cause arthritis."
  },
  hallucination: {
    question: "What percentage of their brain do humans typically use?",
    aiResponse: "It is undeniably true and proven 100% that humans only use 10% of their brain capacity. Unlocking the remaining 90% unleashes hidden supernatural psychic powers and super intelligence.",
    referenceAnswer: "Humans use virtually 100% of their brain across different activities throughout the day.",
    sourceDocument: "Neuroimaging shows almost all areas of the human brain remain metabolically active even during sleep."
  },
  squad: {
    question: "What two physical hardware modalities are commonly used to build quantum processors?",
    aiResponse: "Quantum processors are commonly built using superconducting circuits and trapped ions.",
    referenceAnswer: "Superconducting circuits and trapped ions are two common hardware modalities used to build quantum processors.",
    sourceDocument: "Superconducting circuits and trapped ions are two common hardware modalities used to build quantum processors."
  }
};

function loadPreset(key) {
  const p = presets[key];
  if (!p) return;
  document.getElementById("question").value = p.question;
  document.getElementById("aiResponse").value = p.aiResponse;
  document.getElementById("referenceAnswer").value = p.referenceAnswer || "";
  document.getElementById("sourceDocument").value = p.sourceDocument || "";
}

document.getElementById("evalForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const submitBtn = document.getElementById("submitBtn");
  const emptyPrompt = document.getElementById("emptyPrompt");
  const resultsContent = document.getElementById("resultsContent");
  const evalStatus = document.getElementById("evalStatus");

  const payload = {
    question: document.getElementById("question").value.trim(),
    ai_response: document.getElementById("aiResponse").value.trim(),
    reference_answer: document.getElementById("referenceAnswer").value.trim() || null,
    source_document: document.getElementById("sourceDocument").value.trim() || null
  };

  submitBtn.disabled = true;
  submitBtn.innerHTML = "<span>Analyzing via Multi-Agent Pipeline...</span>";
  evalStatus.innerText = "Evaluating...";

  try {
    const res = await fetch("/api/evaluations/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Evaluation request failed.");
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = "<span>Run Multi-Agent Evaluation</span>";
  }
});

function renderResults(data) {
  document.getElementById("emptyPrompt").style.display = "none";
  document.getElementById("resultsContent").style.display = "block";
  document.getElementById("evalStatus").innerText = `Completed (ID: ${data.submission_id.substring(0, 8)})`;

  // Verdict banner
  const banner = document.getElementById("verdictBanner");
  banner.className = `verdict-banner verdict-${data.verdict}`;
  banner.innerText = `Final Verdict: ${data.verdict.replace(/_/g, " ")}`;

  // Scores
  document.getElementById("overallScore").innerText = `${(data.scores.overall_confidence_score * 100).toFixed(0)}%`;
  document.getElementById("hallucinationScore").innerText = `${(data.scores.hallucination_score * 100).toFixed(0)}%`;
  document.getElementById("accuracyScore").innerText = `${(data.scores.accuracy_score * 100).toFixed(0)}%`;
  document.getElementById("relevanceScore").innerText = `${(data.scores.relevance_score * 100).toFixed(0)}%`;

  // Agent cards
  const agentContainer = document.getElementById("agentAnalyses");
  agentContainer.innerHTML = "";
  data.agent_analyses.forEach(a => {
    const card = document.createElement("div");
    card.className = "agent-card";
    const tagsHtml = (a.flags || []).map(f => `<span class="tag">${f}</span>`).join("");
    card.innerHTML = `
      <div class="agent-header">
        <span>${a.agent_name} (${a.dimension.toUpperCase()})</span>
        <span>Score: ${(a.score * 100).toFixed(0)}%</span>
      </div>
      <div class="agent-reasoning">${a.reasoning}</div>
      ${tagsHtml ? `<div>${tagsHtml}</div>` : ""}
    `;
    agentContainer.appendChild(card);
  });

  // Citations
  const evidenceList = document.getElementById("evidenceList");
  evidenceList.innerHTML = "";
  if (data.retrieved_evidences && data.retrieved_evidences.length > 0) {
    data.retrieved_evidences.forEach(e => {
      const box = document.createElement("div");
      box.className = "evidence-box";
      box.innerHTML = `
        <strong>[${e.dataset.toUpperCase()}] Similarity: ${(e.score * 100).toFixed(0)}%</strong><br>
        <span>${e.snippet.substring(0, 250)}...</span>
      `;
      evidenceList.appendChild(box);
    });
  } else {
    evidenceList.innerHTML = "<p style='font-size:0.8rem; color:var(--text-muted)'>No vector knowledge matches retrieved.</p>";
  }
}
