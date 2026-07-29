"use client";

/**
 * Clickable speaker label with a rename popover.
 *
 * - Type a name (or pick a known voice) and either "Rename here" (this
 *   meeting only) or "Rename & remember" (also saves/updates the voiceprint
 *   so future meetings auto-recognize this person).
 * - Auto-recognized speakers show a subtle ✓ badge and can be corrected or
 *   cleared ("Not X") without hurting the stored voiceprint.
 * - The local mic-channel speaker ("You") is identified by hardware, not by
 *   voice, and is not renameable here.
 */

import { useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { getSpeakerColorClass } from '@/lib/utils';
import { BadgeCheck, Pencil } from 'lucide-react';

export interface SpeakerClusterInfo {
  cluster: number;
  label: string;   // "Speaker 1"
  display: string; // current shown name
  source: 'manual' | 'auto' | 'none';
  suggestion?: { name: string; similarity: number } | null;
  candidates: { name: string; strength: 'strong' | 'possible' | 'weak' }[];
}

const STRENGTH_TEXT: Record<string, string> = {
  strong: 'strong match',
  possible: 'possible match',
};

export function SpeakerNameControl({
  speaker,
  cluster,
  meetingId,
  turnCount,
  knownVoices,
  onChanged,
}: {
  speaker: string;
  cluster: SpeakerClusterInfo | undefined;
  meetingId: string;
  turnCount: number;
  knownVoices: string[];
  onChanged: () => Promise<void> | void;
}) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState('');
  const [busy, setBusy] = useState(false);

  const labelClass = `text-[10px] font-semibold uppercase tracking-wide ${getSpeakerColorClass(speaker)}`;

  // Non-interactive cases: "You" (mic identity), no cluster info (old
  // meeting without speakers.json), or no meeting context.
  if (!cluster || speaker === 'You') {
    return (
      <span className={`block mb-1 ml-[58px] ${labelClass}`} title={speaker === 'You' ? 'Identified from your microphone channel' : undefined}>
        {speaker}
      </span>
    );
  }

  const isNamed = cluster.source !== 'none';
  const suggestion = cluster.suggestion ?? null;

  const rankedVoices = useMemo(() => {
    const byName = new Map(cluster.candidates.map((c) => [c.name.toLowerCase(), c.strength]));
    const named = cluster.candidates.filter((c) => c.strength !== 'weak').map((c) => c.name);
    const rest = knownVoices.filter(
      (n) => !named.some((m) => m.toLowerCase() === n.toLowerCase()) && n.toLowerCase() !== cluster.display.toLowerCase()
    );
    return { named, rest, strengthOf: (n: string) => byName.get(n.toLowerCase()) };
  }, [cluster, knownVoices]);

  const apply = async (name: string, remember: boolean) => {
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    try {
      const changed = await invoke<number>('speaker_rename', {
        meetingId,
        cluster: cluster.cluster,
        name: trimmed,
        remember,
      });
      toast.success(`Renamed ${changed} turn${changed === 1 ? '' : 's'} to ${trimmed}`, {
        description: remember ? 'Voice remembered — future meetings will recognize them.' : undefined,
      });
      setOpen(false);
      setValue('');
      await onChanged();
    } catch (e) {
      toast.error('Rename failed', { description: typeof e === 'string' ? e : undefined });
    } finally {
      setBusy(false);
    }
  };

  const clearName = async () => {
    if (busy) return;
    setBusy(true);
    try {
      await invoke<number>('speaker_clear_name', { meetingId, cluster: cluster.cluster });
      toast.success(`Cleared name — back to ${cluster.label}`);
      setOpen(false);
      await onChanged();
    } catch (e) {
      toast.error('Failed to clear name', { description: typeof e === 'string' ? e : undefined });
    } finally {
      setBusy(false);
    }
  };

  return (
    <span className="block mb-1 ml-[58px]">
      <Popover open={open} onOpenChange={(o) => { setOpen(o); if (o) setValue(''); }}>
        <PopoverTrigger asChild>
          <button
            className={`group inline-flex items-center gap-1 ${labelClass} hover:underline decoration-dotted underline-offset-2 cursor-pointer`}
            title="Rename this speaker"
          >
            {speaker}
            {cluster.source === 'auto' && (
              <BadgeCheck size={11} className="text-emerald-500" aria-label="Recognized automatically" />
            )}
            <Pencil size={9} className="opacity-0 group-hover:opacity-60 transition-opacity" />
          </button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-72 p-3" onOpenAutoFocus={(e) => e.preventDefault()}>
          <div className="space-y-2.5">
            <div className="text-xs text-gray-500">
              {cluster.source === 'auto' ? (
                <>Recognized as <span className="font-medium text-gray-800">{cluster.display}</span> by voice</>
              ) : isNamed ? (
                <>Named <span className="font-medium text-gray-800">{cluster.display}</span> ({cluster.label})</>
              ) : (
                <>{cluster.label} · {turnCount} turn{turnCount === 1 ? '' : 's'}</>
              )}
            </div>

            {suggestion && !isNamed && (
              <button
                className="w-full text-left text-sm px-2.5 py-1.5 rounded-md bg-blue-50 hover:bg-blue-100 text-blue-900 disabled:opacity-50"
                onClick={() => apply(suggestion.name, true)}
                disabled={busy}
              >
                Looks like <span className="font-semibold">{suggestion.name}</span> — confirm?
              </button>
            )}

            <input
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') apply(value, true);
              }}
              placeholder={isNamed ? cluster.display : 'Name this speaker…'}
              className="w-full px-2.5 py-1.5 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />

            {(rankedVoices.named.length > 0 || rankedVoices.rest.length > 0) && (
              <div className="flex flex-wrap gap-1.5">
                {rankedVoices.named.map((n) => (
                  <button
                    key={n}
                    className="px-2 py-0.5 rounded-full text-xs bg-emerald-50 text-emerald-800 hover:bg-emerald-100 disabled:opacity-50"
                    onClick={() => apply(n, true)}
                    disabled={busy}
                    title={STRENGTH_TEXT[rankedVoices.strengthOf(n) ?? ''] ?? undefined}
                  >
                    {n}{rankedVoices.strengthOf(n) === 'strong' ? ' ✓' : ' ?'}
                  </button>
                ))}
                {rankedVoices.rest.map((n) => (
                  <button
                    key={n}
                    className="px-2 py-0.5 rounded-full text-xs bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:opacity-50"
                    onClick={() => apply(n, true)}
                    disabled={busy}
                  >
                    {n}
                  </button>
                ))}
              </div>
            )}

            <div className="flex items-center gap-2 pt-0.5">
              <button
                className="flex-1 px-2 py-1.5 rounded-md text-xs font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                onClick={() => apply(value, true)}
                disabled={busy || !value.trim()}
                title="Renames this meeting and remembers the voice for future meetings"
              >
                Rename &amp; remember
              </button>
              <button
                className="px-2 py-1.5 rounded-md text-xs font-medium bg-gray-100 text-gray-800 hover:bg-gray-200 disabled:opacity-50"
                onClick={() => apply(value, false)}
                disabled={busy || !value.trim()}
                title="Renames only this meeting"
              >
                Only here
              </button>
            </div>

            {isNamed && (
              <button
                className="w-full text-xs text-gray-500 hover:text-red-600 text-left disabled:opacity-50"
                onClick={clearName}
                disabled={busy}
              >
                Not {cluster.display} — clear the name
              </button>
            )}

            <div className="text-[10px] text-gray-400">
              Renames {turnCount} turn{turnCount === 1 ? '' : 's'}. Voice matching runs fully on this Mac.
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </span>
  );
}
