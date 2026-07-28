"use client";

// Live view of a running retranscription: raw transcript blocks stream in
// as they are transcribed (event `retranscription-partial`), progress is
// tracked per stage, and completion triggers a refetch that swaps the
// preview for the final speaker-labeled rows.
//
// Lives outside RetranscribeDialog on purpose: the dialog closes as soon as
// the run starts, so its listeners die — this hook is mounted by the
// transcript panel, which stays visible while the user reads along.

import { useCallback, useEffect, useRef, useState } from 'react';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';

export interface PartialRow {
  text: string;
  audio_start_time: number; // seconds
  audio_end_time: number;
}

export interface StageProgress {
  meeting_id: string;
  stage: string;
  progress_percentage: number;
  message: string;
}

interface StreamState {
  active: boolean;
  rows: PartialRow[];
  progressByStage: Record<string, StageProgress>;
}

const EMPTY: StreamState = { active: false, rows: [], progressByStage: {} };

// Survives navigation between meetings within the session: if the user
// leaves mid-run and comes back, the rows received so far reappear
// immediately instead of waiting for the next event.
const streamCache = new Map<string, StreamState>();

export function useRetranscriptionStream(
  meetingId: string | undefined,
  onComplete?: () => Promise<void> | void
) {
  const [state, setState] = useState<StreamState>(
    () => (meetingId && streamCache.get(meetingId)) || EMPTY
  );
  const onCompleteRef = useRef(onComplete);
  useEffect(() => { onCompleteRef.current = onComplete; }, [onComplete]);

  useEffect(() => {
    if (!meetingId) return;
    setState(streamCache.get(meetingId) || EMPTY);

    const update = (fn: (prev: StreamState) => StreamState) => {
      setState((prev) => {
        const next = fn(prev);
        streamCache.set(meetingId, next);
        return next;
      });
    };

    const unlisteners: UnlistenFn[] = [];
    let cleanedUp = false;

    const setup = async () => {
      const handlers: Array<[string, (payload: any) => void]> = [
        [
          'retranscription-progress',
          (p: StageProgress) =>
            update((prev) => ({
              ...prev,
              active: true,
              progressByStage: { ...prev.progressByStage, [p.stage]: p },
            })),
        ],
        [
          'retranscription-partial',
          (p: PartialRow) =>
            update((prev) => ({
              ...prev,
              active: true,
              rows: [...prev.rows, p].sort(
                (a, b) => a.audio_start_time - b.audio_start_time
              ),
            })),
        ],
        [
          'retranscription-complete',
          async () => {
            streamCache.delete(meetingId);
            toast.success('Retranscription complete');
            // Refetch BEFORE dropping the preview so the list never flashes
            // back to the stale pre-run transcript
            await onCompleteRef.current?.();
            setState(EMPTY);
          },
        ],
        [
          'retranscription-error',
          (p: { error: string }) => {
            streamCache.delete(meetingId);
            setState(EMPTY);
            toast.error(`Retranscription failed: ${p.error}`);
          },
        ],
      ];

      for (const [event, handler] of handlers) {
        const unlisten = await listen<any>(event, (e) => {
          if (e.payload?.meeting_id === meetingId) handler(e.payload);
        });
        if (cleanedUp) {
          unlisten();
          return;
        }
        unlisteners.push(unlisten);
      }
    };
    setup();

    return () => {
      cleanedUp = true;
      unlisteners.forEach((u) => u());
    };
  }, [meetingId]);

  const cancel = useCallback(async () => {
    try {
      await invoke('cancel_retranscription_command');
      if (meetingId) streamCache.delete(meetingId);
      setState(EMPTY);
      toast.info('Retranscription cancelled');
    } catch (err) {
      console.error('Failed to cancel retranscription:', err);
    }
  }, [meetingId]);

  return {
    active: state.active,
    partialRows: state.rows,
    progressByStage: state.progressByStage,
    cancel,
  };
}
