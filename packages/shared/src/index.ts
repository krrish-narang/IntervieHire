export type Severity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type InterviewStatus = 'SCHEDULED' | 'IN_PROGRESS' | 'COMPLETED' | 'EVALUATED' | 'CANCELLED';

export interface ClientToServerTranscript {
  type: 'candidate_transcript';
  sessionId: string;
  text: string;
  timestamp: number;
  latencyMs?: number;
  wpm?: number;
}
export interface ServerToUESpeak {
  type: 'avatar_speak';
  sessionId: string;
  text: string;
  interviewPhase: 'greeting' | 'questioning' | 'follow_up' | 'closing';
  emotionState: 'neutral' | 'encouraging' | 'curious' | 'serious';
}
export interface UEToServerAvatarStatus {
  type: 'avatar_status';
  sessionId: string;
  isSpeaking: boolean;
}
export interface ProctoringPayload {
  type: 'proctoring_event';
  sessionId: string;
  eventType: string;
  severity: Severity;
  metadata: Record<string, unknown>;
  timestamp: number;
}
export interface EvaluationMetric {
  score: number;
  reasoning: string;
}
export interface EvaluationReport {
  answerDepth: EvaluationMetric;
  confidence: EvaluationMetric;
  communication: EvaluationMetric;
  domainKnowledge: EvaluationMetric;
  problemSolving: EvaluationMetric;
  overallScore: number;
  recommendation: 'STRONG_HIRE' | 'HIRE' | 'MAYBE' | 'NO_HIRE';
  strengths: string[];
  risks: string[];
  summary: string;
}
