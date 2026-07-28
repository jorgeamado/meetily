"use client";

import { Transcript, TranscriptSegmentData } from '@/types';
import { TranscriptView } from '@/components/TranscriptView';
import { VirtualizedTranscriptView } from '@/components/VirtualizedTranscriptView';
import { TranscriptButtonGroup } from './TranscriptButtonGroup';
import { useRetranscriptionStream } from './useRetranscriptionStream';
import { Button } from '@/components/ui/button';
import { X } from 'lucide-react';
import { useMemo } from 'react';

const STAGE_LABELS: Record<string, string> = {
  transcribing: 'Transcribing',
  diarizing: 'Identifying speakers',
  saving: 'Finalizing',
  refining: 'AI fix-up',
};

interface TranscriptPanelProps {
  transcripts: Transcript[];
  customPrompt: string;
  onPromptChange: (value: string) => void;
  onCopyTranscript: () => void;
  onOpenMeetingFolder: () => Promise<void>;
  isRecording: boolean;
  disableAutoScroll?: boolean;

  // Optional pagination props (when using virtualization)
  usePagination?: boolean;
  segments?: TranscriptSegmentData[];
  hasMore?: boolean;
  isLoadingMore?: boolean;
  totalCount?: number;
  loadedCount?: number;
  onLoadMore?: () => void;

  // Retranscription props
  meetingId?: string;
  meetingFolderPath?: string | null;
  onRefetchTranscripts?: () => Promise<void>;
}

export function TranscriptPanel({
  transcripts,
  customPrompt,
  onPromptChange,
  onCopyTranscript,
  onOpenMeetingFolder,
  isRecording,
  disableAutoScroll = false,
  usePagination = false,
  segments,
  hasMore,
  isLoadingMore,
  totalCount,
  loadedCount,
  onLoadMore,
  meetingId,
  meetingFolderPath,
  onRefetchTranscripts,
}: TranscriptPanelProps) {
  // Convert transcripts to segments if pagination is not used but we want virtualization
  const convertedSegments = useMemo(() => {
    if (usePagination && segments) {
      return segments;
    }
    // Convert transcripts to segments for virtualization
    return transcripts.map(t => ({
      id: t.id,
      timestamp: t.audio_start_time ?? 0,
      endTime: t.audio_end_time,
      text: t.text,
      confidence: t.confidence,
      speaker: t.speaker,
    }));
  }, [transcripts, usePagination, segments]);

  // Live retranscription: stream raw blocks in as they are transcribed and
  // replace them with the final speaker-labeled rows on completion
  const {
    active: retranscribing,
    partialRows,
    progressByStage,
    cancel: cancelRetranscription,
  } = useRetranscriptionStream(meetingId, onRefetchTranscripts);

  const previewSegments = useMemo(
    () =>
      partialRows.map((r, idx) => ({
        id: `retranscribe-preview-${idx}`,
        timestamp: r.audio_start_time,
        endTime: r.audio_end_time,
        text: r.text,
        confidence: undefined,
        speaker: undefined,
      })),
    [partialRows]
  );
  // Keep showing the existing rows until streamed blocks actually arrive —
  // covers the decode/VAD warm-up and the standalone AI fix-up (which never
  // streams raw blocks, only updates rows at the end)
  const displaySegments =
    retranscribing && previewSegments.length > 0 ? previewSegments : convertedSegments;

  return (
    <div className="hidden md:flex md:w-1/4 lg:w-1/3 min-w-0 border-r border-gray-200 bg-white flex-col relative shrink-0">
      {/* Title area */}
      <div className="p-4 border-b border-gray-200">
        <TranscriptButtonGroup
          transcriptCount={usePagination ? (totalCount ?? convertedSegments.length) : (transcripts?.length || 0)}
          onCopyTranscript={onCopyTranscript}
          onOpenMeetingFolder={onOpenMeetingFolder}
          meetingId={meetingId}
          meetingFolderPath={meetingFolderPath}
          onRefetchTranscripts={onRefetchTranscripts}
        />
      </div>

      {/* Live retranscription progress banner */}
      {retranscribing && (
        <div className="px-4 py-2 border-b border-blue-100 bg-blue-50 space-y-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-blue-900">
              {progressByStage['transcribing']
                ? 'Retranscribing — transcript updates live, speakers appear at the end'
                : 'AI fix-up — improving speaker boundaries and wording'}
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-1.5 text-blue-900"
              onClick={cancelRetranscription}
              title="Cancel retranscription"
            >
              <X size={14} />
            </Button>
          </div>
          {['transcribing', 'diarizing', 'saving']
            .map((stage) => progressByStage[stage])
            .filter(Boolean)
            .map((p) => (
              <div key={p.stage} className="flex items-center gap-2">
                <span className="text-xs text-blue-800 w-36 shrink-0">
                  {STAGE_LABELS[p.stage] ?? p.stage}
                </span>
                <div className="flex-1 bg-blue-100 rounded-full h-1.5">
                  <div
                    className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(p.progress_percentage, 100)}%` }}
                  />
                </div>
                <span className="text-xs text-blue-800 w-8 text-right">
                  {Math.round(p.progress_percentage)}%
                </span>
              </div>
            ))}
        </div>
      )}

      {/* Transcript content - use virtualized view for better performance */}
      <div className="flex-1 overflow-hidden pb-4">
        <VirtualizedTranscriptView
          segments={displaySegments}
          isRecording={isRecording}
          isPaused={false}
          isProcessing={false}
          isStopping={false}
          enableStreaming={false}
          showConfidence={true}
          disableAutoScroll={disableAutoScroll}
          hasMore={retranscribing ? false : hasMore}
          isLoadingMore={retranscribing ? false : isLoadingMore}
          totalCount={retranscribing ? previewSegments.length : totalCount}
          loadedCount={retranscribing ? previewSegments.length : loadedCount}
          onLoadMore={onLoadMore}
        />
      </div>

      {/* Custom prompt input at bottom of transcript section */}
      {!isRecording && convertedSegments.length > 0 && (
        <div className="p-1 border-t border-gray-200">
          <textarea
            placeholder="Add context for AI summary. For example people involved, meeting overview, objective etc..."
            className="w-full px-3 py-2 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 bg-white shadow-sm min-h-[80px] resize-y"
            value={customPrompt}
            onChange={(e) => onPromptChange(e.target.value)}
          />
        </div>
      )}
    </div>
  );
}
