export type WsRole = 'candidate' | 'ue5';
export interface RegisterMessage { type: 'register'; role: WsRole; sessionId: string; }
