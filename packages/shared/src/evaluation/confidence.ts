export interface TranscriptConfidenceAnalysis {
  fillerCount: number;
  hedgeCount: number;
  strongUncertaintyCount: number;
  repeatedWordCount: number;
  totalWords: number;
  fillerRate: number;
  confidenceScore: number;
  confidenceLevel: "high" | "medium" | "low";
  reliability: "high" | "medium" | "low";
  confidencePenalty: number;
  notes: string[];
}

const FILLER_WORDS = new Set([
  "um",
  "umm",
  "uh",
  "uhh",
  "ah",
  "ahh",
  "er",
  "erm",
  "hmm",
]);

const MILD_HEDGE_PHRASES = [
  "i guess",
  "sort of",
  "kind of",
  "maybe",
  "i feel like",
];

const STRONG_UNCERTAINTY_PHRASES = [
  "i don't know",
  "i do not know",
  "not sure",
  "i have no idea",
  "i can't remember",
  "i cannot remember",
];

export function analyzeTranscriptConfidence(transcript: string): TranscriptConfidenceAnalysis {
  const normalized = transcript.toLowerCase().replace(/[’]/g, "'");
  const words = normalized.match(/[a-z']+/g) ?? [];
  const totalWords = words.length;
  const fillerCount = words.filter((word) => FILLER_WORDS.has(word)).length;
  const hedgeCount = countPhrases(normalized, MILD_HEDGE_PHRASES);
  const strongUncertaintyCount = countPhrases(normalized, STRONG_UNCERTAINTY_PHRASES);
  const repeatedWordCount = words.reduce((count, word, index) => {
    if (index === 0) return count;
    return word === words[index - 1] && !FILLER_WORDS.has(word) ? count + 1 : count;
  }, 0);
  const fillerRate = totalWords === 0 ? 0 : fillerCount / totalWords;
  const confidenceScore = calculateConfidenceScore({
    fillerCount,
    hedgeCount,
    strongUncertaintyCount,
    repeatedWordCount,
    totalWords,
    fillerRate,
  });
  const confidenceLevel = getConfidenceLevel(confidenceScore);
  const reliability = getReliability(totalWords);
  const confidencePenalty = calculateCommunicationPenalty(confidenceScore, reliability);
  const notes = buildConfidenceNotes({
    fillerCount,
    hedgeCount,
    strongUncertaintyCount,
    repeatedWordCount,
    totalWords,
    confidenceScore,
    reliability,
  });

  return {
    fillerCount,
    hedgeCount,
    strongUncertaintyCount,
    repeatedWordCount,
    totalWords,
    fillerRate,
    confidenceScore,
    confidenceLevel,
    reliability,
    confidencePenalty,
    notes,
  };
}

function calculateConfidenceScore(params: {
  fillerCount: number;
  hedgeCount: number;
  strongUncertaintyCount: number;
  repeatedWordCount: number;
  totalWords: number;
  fillerRate: number;
}): number {
  if (params.totalWords === 0) {
    return 0;
  }

  const fillerPenalty = params.fillerRate > 0.1 ? 20 : params.fillerRate > 0.06 ? 12 : params.fillerRate > 0.03 ? 6 : 0;
  const hedgePenalty = Math.min(15, params.hedgeCount * 5);
  const uncertaintyPenalty = Math.min(45, params.strongUncertaintyCount * 18);
  // Consecutive duplicates are often speech-to-text artifacts, so keep their effect small.
  const repeatPenalty = Math.min(6, params.repeatedWordCount * 2);
  const shortTranscriptAdjustment = params.totalWords < 8 ? -10 : 0;

  return clampScore(78 + shortTranscriptAdjustment - fillerPenalty - hedgePenalty - uncertaintyPenalty - repeatPenalty);
}

function calculateCommunicationPenalty(
  confidenceScore: number,
  reliability: TranscriptConfidenceAnalysis["reliability"],
): number {
  if (reliability === "low") return 0;
  if (confidenceScore < 35) return 6;
  if (confidenceScore < 50) return 4;
  if (confidenceScore < 65) return 2;
  return 0;
}

function buildConfidenceNotes(params: {
  fillerCount: number;
  hedgeCount: number;
  strongUncertaintyCount: number;
  repeatedWordCount: number;
  totalWords: number;
  confidenceScore: number;
  reliability: TranscriptConfidenceAnalysis["reliability"];
}): string[] {
  const notes: string[] = [];

  if (params.totalWords === 0) {
    return ["No transcript words were available; expressed confidence could not be assessed."];
  }

  if (params.strongUncertaintyCount > 0) {
    notes.push(`${params.strongUncertaintyCount} explicit uncertainty phrase${params.strongUncertaintyCount === 1 ? "" : "s"} detected.`);
  }

  if (params.fillerCount > 0) {
    notes.push(`${params.fillerCount} filler word${params.fillerCount === 1 ? "" : "s"} detected.`);
  }

  if (params.hedgeCount > 0) {
    notes.push(`${params.hedgeCount} hedging phrase${params.hedgeCount === 1 ? "" : "s"} detected.`);
  }

  if (params.repeatedWordCount > 0) {
    notes.push(`${params.repeatedWordCount} repeated word pattern${params.repeatedWordCount === 1 ? "" : "s"} detected; these may be transcription artifacts.`);
  }

  if (params.confidenceScore >= 65 && notes.length === 0) {
    notes.push("No material textual hesitation or uncertainty markers were detected.");
  }

  notes.push(`Assessment reliability is ${params.reliability}; transcript text cannot measure tone, pace, volume, or body language.`);

  return notes;
}

function countPhrases(text: string, phrases: string[]): number {
  return phrases.reduce((count, phrase) => {
    const matches = text.match(new RegExp(`\\b${escapeRegExp(phrase)}\\b`, "g"));
    return count + (matches?.length ?? 0);
  }, 0);
}

function getConfidenceLevel(score: number): TranscriptConfidenceAnalysis["confidenceLevel"] {
  if (score >= 75) return "high";
  if (score >= 50) return "medium";
  return "low";
}

function getReliability(totalWords: number): TranscriptConfidenceAnalysis["reliability"] {
  if (totalWords >= 40) return "high";
  if (totalWords >= 15) return "medium";
  return "low";
}

function clampScore(score: number): number {
  return Math.max(0, Math.min(100, Math.round(score)));
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
