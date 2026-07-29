"use client";

/**
 * Settings section listing remembered voices (speaker naming library).
 * Voices are added by renaming a speaker in a meeting with
 * "Rename & remember"; deleting one only stops future auto-recognition —
 * already-named meetings keep their names.
 */

import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import { Trash2, UserRound } from 'lucide-react';

interface VoiceInfo {
  name: string;
  model: string;
  cleanSecs: number;
  meetings: number;
  updatedAt: string;
}

function formatSpeech(secs: number): string {
  if (secs >= 90) return `${Math.round(secs / 60)} min of speech`;
  return `${Math.round(secs)}s of speech`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? '' : d.toLocaleDateString();
}

export function KnownVoices() {
  const [voices, setVoices] = useState<VoiceInfo[] | null>(null);

  const refresh = useCallback(async () => {
    try {
      setVoices(await invoke<VoiceInfo[]>('voices_list'));
    } catch {
      setVoices([]);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const remove = async (v: VoiceInfo) => {
    try {
      await invoke('voice_delete', { name: v.name, model: v.model });
      toast.success(`Forgot ${v.name}'s voice`, {
        description: 'Existing meetings keep their names; future meetings will no longer auto-recognize them.',
      });
      await refresh();
    } catch (e) {
      toast.error('Failed to delete voice', { description: typeof e === 'string' ? e : undefined });
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h3 className="text-lg font-semibold mb-1">Known voices</h3>
      <p className="text-sm text-gray-500 mb-4">
        People you have named with &ldquo;Rename &amp; remember&rdquo;. New meetings are matched
        against these voices locally — nothing leaves this Mac.
      </p>

      {voices === null ? (
        <p className="text-sm text-gray-400">Loading…</p>
      ) : voices.length === 0 ? (
        <p className="text-sm text-gray-400">
          No voices yet. Open a meeting, click a speaker label, and choose
          &ldquo;Rename &amp; remember&rdquo; to teach the first one.
        </p>
      ) : (
        <ul className="divide-y divide-gray-100">
          {voices.map((v) => (
            <li key={`${v.model}:${v.name}`} className="flex items-center gap-3 py-2.5">
              <UserRound size={18} className="text-gray-400 shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-gray-900">{v.name}</div>
                <div className="text-xs text-gray-500">
                  {v.meetings} meeting{v.meetings === 1 ? '' : 's'} · {formatSpeech(v.cleanSecs)}
                  {formatDate(v.updatedAt) && <> · updated {formatDate(v.updatedAt)}</>}
                </div>
              </div>
              <button
                className="p-1.5 rounded-md text-gray-400 hover:text-red-600 hover:bg-red-50"
                onClick={() => remove(v)}
                title={`Forget ${v.name}'s voice`}
              >
                <Trash2 size={15} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
