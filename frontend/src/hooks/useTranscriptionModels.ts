import { useState, useCallback, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';

export interface RawModelInfo {
  name: string;
  size_mb: number;
  status: 'Available' | 'Missing' | { Downloading: { progress: number } } | { Error: string };
}

export interface ModelOption {
  provider: 'whisper' | 'parakeet' | 'fluidAudio';
  name: string;
  displayName: string;
  size_mb: number;
}

function getFluidDisplayName(modelName: string): string {
  if (modelName === 'fluid-parakeet-v3') {
    return '🍎 Parakeet (Neural Engine): v3';
  }
  if (modelName === 'fluid-parakeet-ultra') {
    return '🍎 Parakeet (Neural Engine): Ultra';
  }
  return `🍎 Parakeet (Neural Engine): ${modelName}`;
}

interface TranscriptModelConfig {
  provider?: string;
  model?: string;
}

/**
 * Custom hook for fetching and managing transcription models (Whisper and Parakeet).
 *
 * This hook centralizes the model fetching logic that was previously duplicated
 * in ImportAudioDialog and RetranscribeDialog components.
 *
 * @param transcriptModelConfig - User's saved model configuration from context
 * @returns Object containing available models, selected model key, loading state, and fetch function
 */
export function useTranscriptionModels(transcriptModelConfig: TranscriptModelConfig | undefined) {
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [selectedModelKey, setSelectedModelKey] = useState<string>('');
  const [loadingModels, setLoadingModels] = useState(false);
  // Track whether the user has manually changed the model selection
  const userSelectedRef = useRef(false);

  // Wrap setSelectedModelKey to track user-initiated changes
  const setSelectedModelKeyWithTracking = useCallback((key: string) => {
    userSelectedRef.current = true;
    setSelectedModelKey(key);
  }, []);

  const fetchModels = useCallback(async () => {
    setLoadingModels(true);
    const allModels: ModelOption[] = [];

    // Fetch Whisper models
    try {
      const whisperModels = await invoke<RawModelInfo[]>('whisper_get_available_models');
      const availableWhisper = whisperModels
        .filter((m) => m.status === 'Available')
        .map((m) => ({
          provider: 'whisper' as const,
          name: m.name,
          displayName: `🏠 Whisper: ${m.name}`,
          size_mb: m.size_mb,
        }));
      allModels.push(...availableWhisper);
    } catch (err) {
      console.error('Failed to fetch Whisper models:', err);
    }

    // Fetch Parakeet models
    try {
      const parakeetModels = await invoke<RawModelInfo[]>('parakeet_get_available_models');
      const availableParakeet = parakeetModels
        .filter((m) => m.status === 'Available')
        .map((m) => ({
          provider: 'parakeet' as const,
          name: m.name,
          displayName: `⚡ Parakeet: ${m.name}`,
          size_mb: m.size_mb,
        }));
      allModels.push(...availableParakeet);
    } catch (err) {
      console.error('Failed to fetch Parakeet models:', err);
    }

    // Fetch FluidAudio (Apple Neural Engine) models when available on macOS
    try {
      const fluidAvailable = await invoke<boolean>('fluid_is_available');
      if (fluidAvailable) {
        const fluidModels = await invoke<RawModelInfo[]>('fluid_get_available_models');
        const availableFluid = fluidModels
          .filter((m) => m.status === 'Available')
          .map((m) => ({
            provider: 'fluidAudio' as const,
            name: m.name,
            displayName: getFluidDisplayName(m.name),
            size_mb: m.size_mb,
          }));
        allModels.push(...availableFluid);
      }
    } catch (err) {
      console.error('Failed to fetch FluidAudio models:', err);
    }

    setAvailableModels(allModels);

    // Set default model based on user's saved configuration
    const configuredProvider = transcriptModelConfig?.provider || '';
    const configuredModel = transcriptModelConfig?.model || '';

    // Try to match the configured model
    // Note: 'localWhisper' in config maps to 'whisper' provider in model list
    const configuredMatch = allModels.find(
      (m) =>
        (configuredProvider === 'localWhisper' && m.provider === 'whisper' && m.name === configuredModel) ||
        (configuredProvider === 'parakeet' && m.provider === 'parakeet' && m.name === configuredModel) ||
        (configuredProvider === 'fluidAudio' && m.provider === 'fluidAudio' && m.name === configuredModel)
    );

    // Only set default model if user hasn't manually selected one.
    // FluidAudio models never become the auto-selected fallback default unless explicitly configured.
    if (!userSelectedRef.current) {
      if (configuredMatch) {
        // Use the configured model if available
        setSelectedModelKey(`${configuredMatch.provider}:${configuredMatch.name}`);
      } else {
        const defaultModel = allModels.find((m) => m.provider !== 'fluidAudio');
        if (defaultModel) {
          // Fall back to first available non-FluidAudio model
          setSelectedModelKey(`${defaultModel.provider}:${defaultModel.name}`);
        }
      }
    }

    setLoadingModels(false);
  }, [transcriptModelConfig]);

  // Reset user selection tracking (call when dialog opens fresh)
  const resetSelection = useCallback(() => {
    userSelectedRef.current = false;
  }, []);

  return {
    availableModels,
    selectedModelKey,
    setSelectedModelKey: setSelectedModelKeyWithTracking,
    loadingModels,
    fetchModels,
    resetSelection,
  };
}
