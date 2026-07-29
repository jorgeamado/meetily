'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import Analytics from '@/lib/analytics';
import { invoke } from '@tauri-apps/api/core';
import { useRecordingState } from '@/contexts/RecordingStateContext';


interface SidebarItem {
  id: string;
  title: string;
  type: 'folder' | 'file';
  hasTranscripts?: boolean;
  durationSeconds?: number | null;
  createdAt?: string;
  children?: SidebarItem[];
}

export interface CurrentMeeting {
  id: string;
  title: string;
  has_transcripts?: boolean;
  duration_seconds?: number | null;
  created_at?: string;
  folder_id?: string | null;
}

export interface MeetingFolder {
  id: string;
  title: string;
}

/// Live processing state for a meeting (transcription/retranscription/AI fix-up)
export interface MeetingActivity {
  stage: string;
  progress: number;
  message: string;
}

// Search result type for transcript search
interface TranscriptSearchResult {
  id: string;
  title: string;
  matchContext: string;
  timestamp: string;
};

interface SidebarContextType {
  currentMeeting: CurrentMeeting | null;
  setCurrentMeeting: (meeting: CurrentMeeting | null) => void;
  sidebarItems: SidebarItem[];
  isCollapsed: boolean;
  toggleCollapse: () => void;
  meetings: CurrentMeeting[];
  setMeetings: (meetings: CurrentMeeting[]) => void;
  isMeetingActive: boolean;
  setIsMeetingActive: (active: boolean) => void;
  handleRecordingToggle: () => void;
  searchTranscripts: (query: string) => Promise<void>;
  searchResults: TranscriptSearchResult[];
  isSearching: boolean;
  setServerAddress: (address: string) => void;
  serverAddress: string;
  transcriptServerAddress: string;
  setTranscriptServerAddress: (address: string) => void;
  // Summary polling management
  activeSummaryPolls: Map<string, NodeJS.Timeout>;
  startSummaryPolling: (meetingId: string, processId: string, onUpdate: (result: any) => void) => void;
  stopSummaryPolling: (meetingId: string) => void;
  // Refetch meetings from backend
  refetchMeetings: () => Promise<void>;
  // Per-meeting live processing state (keyed by meeting id)
  meetingActivity: Map<string, MeetingActivity>;
  // Meetings whose background run finished and hasn't been looked at yet
  recentlyCompleted: Set<string>;
  // Clear the completion marker (called when the meeting is opened)
  acknowledgeCompletion: (meetingId: string) => void;
  // User folders for organizing meetings
  folders: MeetingFolder[];
  createFolder: (title: string) => Promise<MeetingFolder | null>;
  renameFolder: (folderId: string, title: string) => Promise<boolean>;
  deleteFolder: (folderId: string) => Promise<boolean>;
  moveMeetingToFolder: (meetingId: string, folderId: string | null) => Promise<true | string>;
}

const SidebarContext = createContext<SidebarContextType | null>(null);

export const useSidebar = () => {
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error('useSidebar must be used within a SidebarProvider');
  }
  return context;
};

export function SidebarProvider({ children }: { children: React.ReactNode }) {
  const [currentMeeting, setCurrentMeeting] = useState<CurrentMeeting | null>({ id: 'intro-call', title: '+ New Call' });
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [meetings, setMeetings] = useState<CurrentMeeting[]>([]);
  const [sidebarItems, setSidebarItems] = useState<SidebarItem[]>([]);
  const [isMeetingActive, setIsMeetingActive] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [serverAddress, setServerAddress] = useState('');
  const [transcriptServerAddress, setTranscriptServerAddress] = useState('');
  const [activeSummaryPolls, setActiveSummaryPolls] = useState<Map<string, NodeJS.Timeout>>(new Map());

  // Use recording state from RecordingStateContext (single source of truth)
  const { isRecording } = useRecordingState();

  const pathname = usePathname();
  const router = useRouter();

  const [folders, setFolders] = useState<MeetingFolder[]>([]);

  // Extract fetchMeetings as a reusable function (folders come along — the
  // sidebar tree needs both to stay consistent)
  const fetchMeetings = React.useCallback(async () => {
    if (serverAddress) {
      try {
        const [meetings, folderList] = await Promise.all([
          invoke('api_get_meetings') as Promise<Array<CurrentMeeting>>,
          (invoke('api_list_folders') as Promise<MeetingFolder[]>).catch(() => [] as MeetingFolder[]),
        ]);
        const transformedMeetings = meetings.map((meeting: any) => ({
          id: meeting.id,
          title: meeting.title,
          has_transcripts: meeting.has_transcripts,
          duration_seconds: meeting.duration_seconds,
          created_at: meeting.created_at,
          folder_id: meeting.folder_id
        }));
        setMeetings(transformedMeetings);
        setFolders(folderList);
        Analytics.trackBackendConnection(true);
      } catch (error) {
        console.error('Error fetching meetings:', error);
        setMeetings([]);
        Analytics.trackBackendConnection(false, error instanceof Error ? error.message : 'Unknown error');
      }
    }
  }, [serverAddress]);

  const createFolder = React.useCallback(async (title: string): Promise<MeetingFolder | null> => {
    try {
      const folder = await invoke('api_create_folder', { title }) as MeetingFolder;
      setFolders(prev => [...prev, folder].sort((a, b) => a.title.localeCompare(b.title)));
      return folder;
    } catch (error) {
      console.error('Failed to create folder:', error);
      return null;
    }
  }, []);

  const renameFolder = React.useCallback(async (folderId: string, title: string) => {
    try {
      await invoke('api_rename_folder', { folderId, title });
      setFolders(prev => prev.map(f => f.id === folderId ? { ...f, title } : f));
      return true;
    } catch (error) {
      console.error('Failed to rename folder:', error);
      return false;
    }
  }, []);

  const deleteFolder = React.useCallback(async (folderId: string) => {
    try {
      await invoke('api_delete_folder', { folderId });
      setFolders(prev => prev.filter(f => f.id !== folderId));
      setMeetings(prev => prev.map(m => m.folder_id === folderId ? { ...m, folder_id: null } : m));
      return true;
    } catch (error) {
      console.error('Failed to delete folder:', error);
      return false;
    }
  }, []);

  const moveMeetingToFolder = React.useCallback(async (meetingId: string, folderId: string | null) => {
    try {
      await invoke('api_set_meeting_folder', { meetingId, folderId });
      setMeetings(prev => prev.map(m => m.id === meetingId ? { ...m, folder_id: folderId } : m));
      return true;
    } catch (error) {
      console.error('Failed to move meeting:', error);
      return typeof error === 'string' ? error : 'Failed to move meeting';
    }
  }, []);

  // Track background transcription/fix-up runs app-wide so the meeting list
  // can show progress and completion regardless of which page is open.
  const [meetingActivity, setMeetingActivity] = useState<Map<string, MeetingActivity>>(new Map());
  const [recentlyCompleted, setRecentlyCompleted] = useState<Set<string>>(new Set());

  const acknowledgeCompletion = React.useCallback((meetingId: string) => {
    setRecentlyCompleted(prev => {
      if (!prev.has(meetingId)) return prev;
      const next = new Set(prev);
      next.delete(meetingId);
      return next;
    });
  }, []);

  useEffect(() => {
    const unlisteners: Array<() => void> = [];
    let cancelled = false;

    const setup = async () => {
      const { listen } = await import('@tauri-apps/api/event');

      const finish = (meetingId: string, ok: boolean) => {
        setMeetingActivity(prev => {
          const next = new Map(prev);
          next.delete(meetingId);
          return next;
        });
        if (ok) {
          // Acknowledged state, not a timer: the check stays until the user
          // opens the meeting, so a run finishing while they look away is
          // still noticeable.
          setRecentlyCompleted(prev => new Set(prev).add(meetingId));
          // Durations and has_transcripts may have changed
          fetchMeetings();
        }
      };

      const uProgress = await listen<any>('retranscription-progress', (event) => {
        const { meeting_id, stage, progress_percentage, message } = event.payload;
        setMeetingActivity(prev => new Map(prev).set(meeting_id, {
          stage,
          progress: progress_percentage,
          message
        }));
      });
      const uComplete = await listen<any>('retranscription-complete', (event) => {
        finish(event.payload.meeting_id, true);
      });
      const uError = await listen<any>('retranscription-error', (event) => {
        finish(event.payload.meeting_id, false);
      });

      if (cancelled) {
        uProgress(); uComplete(); uError();
        return;
      }
      unlisteners.push(uProgress, uComplete, uError);
    };
    setup();

    return () => {
      cancelled = true;
      unlisteners.forEach(u => u());
    };
  }, [fetchMeetings]);

  useEffect(() => {
    fetchMeetings();
  }, [serverAddress, fetchMeetings]);

  useEffect(() => {
    const fetchSettings = async () => {
      setServerAddress('http://localhost:5167');
      setTranscriptServerAddress('http://127.0.0.1:8178/stream');
    };
    fetchSettings();
  }, []);

  const meetingToItem = (meeting: CurrentMeeting): SidebarItem => ({
    id: meeting.id,
    title: meeting.title,
    type: 'file' as const,
    hasTranscripts: meeting.has_transcripts !== false,
    durationSeconds: meeting.duration_seconds ?? null,
    createdAt: meeting.created_at
  });

  // Newest first (backend already sorts, but folder grouping re-partitions)
  const byDateDesc = (a: CurrentMeeting, b: CurrentMeeting) =>
    (b.created_at ?? '').localeCompare(a.created_at ?? '');

  const folderNodes: SidebarItem[] = folders
    .map(folder => {
      const inFolder = meetings.filter(m => m.folder_id === folder.id).sort(byDateDesc);
      return {
        id: folder.id,
        title: folder.title,
        type: 'folder' as const,
        children: inFolder.map(meetingToItem)
      };
    })
    // Folders with recent activity first; empty folders last, alphabetical
    .sort((a, b) => {
      const newest = (n: SidebarItem) => n.children?.[0]?.createdAt ?? '';
      return newest(b).localeCompare(newest(a)) || a.title.localeCompare(b.title);
    });

  const ungrouped = meetings
    .filter(m => !m.folder_id || !folders.some(f => f.id === m.folder_id))
    .sort(byDateDesc)
    .map(meetingToItem);

  const baseItems: SidebarItem[] = [
    {
      id: 'meetings',
      title: 'Meeting Notes',
      type: 'folder' as const,
      children: [...folderNodes, ...ungrouped]
    },
  ];


  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  // Update current meeting when on home page
  useEffect(() => {
    if (pathname === '/') {
      setCurrentMeeting({ id: 'intro-call', title: '+ New Call' });
    }
    setSidebarItems(baseItems);
  }, [pathname]);

  // Update sidebar items when meetings or folders change
  useEffect(() => {
    setSidebarItems(baseItems);
  }, [meetings, folders]);

  // Function to handle recording toggle from sidebar
  const handleRecordingToggle = () => {
    if (!isRecording) {
      // Check if already on home page
      if (pathname === '/') {
        // Already on home - trigger recording directly via custom event
        console.log('Triggering recording from sidebar (already on home page)');
        window.dispatchEvent(new CustomEvent('start-recording-from-sidebar'));
      } else {
        // Not on home - navigate and use auto-start mechanism
        console.log('Navigating to home page with auto-start flag');
        sessionStorage.setItem('autoStartRecording', 'true');
        router.push('/');
      }

      // Track recording initiation from sidebar
      Analytics.trackButtonClick('start_recording', 'sidebar');
    }
    // The actual recording start/stop is handled in the Home component
  };

  // Function to search through meeting transcripts
  const searchTranscripts = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      setIsSearching(true);


      const results = await invoke('api_search_transcripts', { query }) as TranscriptSearchResult[];
      setSearchResults(results);
    } catch (error) {
      console.error('Error searching transcripts:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  // Summary polling management
  const startSummaryPolling = React.useCallback((
    meetingId: string,
    processId: string,
    onUpdate: (result: any) => void
  ) => {
    // Stop existing poll for this meeting if any
    if (activeSummaryPolls.has(meetingId)) {
      clearInterval(activeSummaryPolls.get(meetingId)!);
    }

    console.log(`📊 Starting polling for meeting ${meetingId}, process ${processId}`);

    let pollCount = 0;
    const MAX_POLLS = 200; // ~16.5 minutes at 5-second intervals (slightly longer than backend's 15-min timeout to avoid race conditions)

    const pollInterval = setInterval(async () => {
      pollCount++;

      // Timeout safety: Stop after 10 minutes
      if (pollCount >= MAX_POLLS) {
        console.warn(`⏱️ Polling timeout for ${meetingId} after ${MAX_POLLS} iterations`);
        clearInterval(pollInterval);
        setActiveSummaryPolls(prev => {
          const next = new Map(prev);
          next.delete(meetingId);
          return next;
        });
        onUpdate({
          status: 'error',
          error: 'Summary generation timed out after 15 minutes. Please try again or check your model configuration.'
        });
        return;
      }
      try {
        const result = await invoke('api_get_summary', {
          meetingId: meetingId,
        }) as any;

        console.log(`📊 Polling update for ${meetingId}:`, result.status);

        // Call the update callback with result
        onUpdate(result);

        // Stop polling if completed, error, failed, cancelled, or idle (after initial processing)
        if (result.status === 'completed' || result.status === 'error' || result.status === 'failed' || result.status === 'cancelled') {
          console.log(`Polling completed for ${meetingId}, status: ${result.status}`);
          clearInterval(pollInterval);
          setActiveSummaryPolls(prev => {
            const next = new Map(prev);
            next.delete(meetingId);
            return next;
          });
        } else if (result.status === 'idle' && pollCount > 1) {
          // If we get 'idle' after polling started, process completed/disappeared
          console.log(`Process completed or not found for ${meetingId}, stopping poll`);
          clearInterval(pollInterval);
          setActiveSummaryPolls(prev => {
            const next = new Map(prev);
            next.delete(meetingId);
            return next;
          });
        }
      } catch (error) {
        console.error(`Polling error for ${meetingId}:`, error);
        // Report error to callback
        onUpdate({
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error'
        });
        clearInterval(pollInterval);
        setActiveSummaryPolls(prev => {
          const next = new Map(prev);
          next.delete(meetingId);
          return next;
        });
      }
    }, 5000); // Poll every 5 seconds

    setActiveSummaryPolls(prev => new Map(prev).set(meetingId, pollInterval));
  }, [activeSummaryPolls]);

  const stopSummaryPolling = React.useCallback((meetingId: string) => {
    const pollInterval = activeSummaryPolls.get(meetingId);
    if (pollInterval) {
      console.log(`⏹️ Stopping polling for meeting ${meetingId}`);
      clearInterval(pollInterval);
      setActiveSummaryPolls(prev => {
        const next = new Map(prev);
        next.delete(meetingId);
        return next;
      });
    }
  }, [activeSummaryPolls]);

  // Cleanup all polling intervals on unmount
  useEffect(() => {
    return () => {
      console.log('🧹 Cleaning up all summary polling intervals');
      activeSummaryPolls.forEach(interval => clearInterval(interval));
    };
  }, [activeSummaryPolls]);



  return (
    <SidebarContext.Provider value={{
      currentMeeting,
      setCurrentMeeting,
      sidebarItems,
      isCollapsed,
      toggleCollapse,
      meetings,
      setMeetings,
      isMeetingActive,
      setIsMeetingActive,
      handleRecordingToggle,
      searchTranscripts,
      searchResults,
      isSearching,
      setServerAddress,
      serverAddress,
      transcriptServerAddress,
      setTranscriptServerAddress,
      activeSummaryPolls,
      startSummaryPolling,
      stopSummaryPolling,
      refetchMeetings: fetchMeetings,
      meetingActivity,
      recentlyCompleted,
      acknowledgeCompletion,
      folders,
      createFolder,
      renameFolder,
      deleteFolder,
      moveMeetingToFolder,
    }}>
      {children}
    </SidebarContext.Provider>
  );
}
