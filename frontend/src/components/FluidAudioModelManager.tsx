import React, { useState, useEffect, useRef, useCallback } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { ModelStatus, formatFileSize } from '../lib/parakeet';
import { ConfirmationModal } from './ConfirmationModel/confirmation-modal';

export type FluidLoadStage = 'download' | 'compile';

export interface FluidModelInfo {
  name: string;
  display_name: string;
  size_mb: number;
  description: string;
  status: ModelStatus;
}

interface FluidDownloadProgressEvent {
  modelName: string;
  stage: FluidLoadStage;
  progress: number;
}

interface FluidAudioModelManagerProps {
  selectedModel?: string;
  onModelSelect?: (modelName: string) => void;
  className?: string;
  autoSave?: boolean;
}

export function FluidAudioModelManager({
  selectedModel,
  onModelSelect,
  className = '',
  autoSave = false
}: FluidAudioModelManagerProps) {
  const [models, setModels] = useState<FluidModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [initialized, setInitialized] = useState(false);
  const [listenersReady, setListenersReady] = useState(false);
  const [listenerRetryCount, setListenerRetryCount] = useState(0);
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());
  const [deletingModels, setDeletingModels] = useState<Set<string>>(new Set());
  const [modelPendingDelete, setModelPendingDelete] = useState<FluidModelInfo | null>(null);
  const [stageByModel, setStageByModel] = useState<Record<string, FluidLoadStage>>({});

  const onModelSelectRef = useRef(onModelSelect);
  const autoSaveRef = useRef(autoSave);
  const latestStatusByModelRef = useRef<Map<string, ModelStatus>>(new Map());
  const latestStageByModelRef = useRef<Map<string, FluidLoadStage>>(new Map());
  const modelsRef = useRef<FluidModelInfo[]>([]);

  useEffect(() => {
    onModelSelectRef.current = onModelSelect;
    autoSaveRef.current = autoSave;
  }, [onModelSelect, autoSave]);

  useEffect(() => {
    modelsRef.current = models;
  }, [models]);

  const saveModelSelection = useCallback(async (modelName: string) => {
    try {
      await invoke('api_save_transcript_config', {
        provider: 'fluidAudio',
        model: modelName,
        apiKey: null
      });
    } catch (err) {
      console.error('Failed to save FluidAudio model selection:', err);
    }
  }, []);

  const getDisplayName = useCallback((modelName: string): string => {
    const found = modelsRef.current.find(m => m.name === modelName);
    if (found?.display_name) return found.display_name;
    if (modelName === 'fluid-parakeet-v3') return 'Parakeet v3';
    if (modelName === 'fluid-parakeet-ultra') return 'Parakeet Ultra';
    return modelName;
  }, []);

  const clearLocalDownloadingState = useCallback((modelName: string) => {
    const latestStatus = latestStatusByModelRef.current.get(modelName);
    if (latestStatus && typeof latestStatus === 'object' && 'Downloading' in latestStatus) {
      latestStatusByModelRef.current.delete(modelName);
    }
    latestStageByModelRef.current.delete(modelName);

    setStageByModel(prev => {
      if (!(modelName in prev)) return prev;
      const next = { ...prev };
      delete next[modelName];
      return next;
    });
    setDownloadingModels(prev => {
      if (!prev.has(modelName)) return prev;
      const next = new Set(prev);
      next.delete(modelName);
      return next;
    });
  }, []);

  const fetchModels = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const modelList = await invoke<FluidModelInfo[]>('fluid_get_available_models');
      setModels(
        modelList.map(model => ({
          ...model,
          status: latestStatusByModelRef.current.get(model.name) ?? model.status
        }))
      );
      setInitialized(true);
    } catch (err) {
      console.error('Failed to load FluidAudio models:', err);
      const message = err instanceof Error ? err.message : typeof err === 'string' ? err : 'Failed to load models';
      setError(message);
      toast.error('Failed to load Neural Engine models', {
        description: message,
        duration: 5000
      });
    } finally {
      setLoading(false);
    }
  }, []);

  // Set up event listeners before first fetch and clean up on unmount
  useEffect(() => {
    let disposed = false;
    let registeredUnlisteners: Array<() => void> = [];

    const setupListeners = async () => {
      const registrations = await Promise.allSettled([
        listen<FluidDownloadProgressEvent>(
          'fluid-model-download-progress',
          (event) => {
            const { modelName, stage, progress } = event.payload;
            const modelStatus: ModelStatus = { Downloading: { progress } };
            latestStatusByModelRef.current.set(modelName, modelStatus);
            latestStageByModelRef.current.set(modelName, stage);

            setStageByModel(prev => (
              prev[modelName] === stage ? prev : { ...prev, [modelName]: stage }
            ));
            setDownloadingModels(prev => {
              if (prev.has(modelName)) return prev;
              const next = new Set(prev);
              next.add(modelName);
              return next;
            });
            setModels(prevModels =>
              prevModels.map(model =>
                model.name === modelName
                  ? { ...model, status: modelStatus }
                  : model
              )
            );
          }
        ),
        listen<{ modelName: string }>(
          'fluid-model-download-complete',
          (event) => {
            const { modelName } = event.payload;
            const displayName = getDisplayName(modelName);
            latestStatusByModelRef.current.set(modelName, 'Available');
            clearLocalDownloadingState(modelName);

            setModels(prevModels =>
              prevModels.map(model =>
                model.name === modelName
                  ? { ...model, status: 'Available' as ModelStatus }
                  : model
              )
            );

            toast.success(`🍎 ${displayName} ready!`, {
              description: 'Model downloaded and prepared for the Apple Neural Engine',
              duration: 4000
            });

            if (onModelSelectRef.current) {
              onModelSelectRef.current(modelName);
              if (autoSaveRef.current) {
                void saveModelSelection(modelName);
              }
            }
          }
        ),
        listen<{ modelName: string; error: string }>(
          'fluid-model-download-error',
          (event) => {
            const { modelName, error: errMsg } = event.payload;
            const displayName = getDisplayName(modelName);
            latestStatusByModelRef.current.set(modelName, { Error: errMsg });
            clearLocalDownloadingState(modelName);

            setModels(prevModels =>
              prevModels.map(model =>
                model.name === modelName
                  ? { ...model, status: { Error: errMsg } as ModelStatus }
                  : model
              )
            );

            toast.error(`Failed to prepare ${displayName}`, {
              description: errMsg,
              duration: 6000,
              action: {
                label: 'Retry',
                onClick: () => downloadModel(modelName)
              }
            });
          }
        )
      ]);

      const unlisteners = registrations.flatMap(result =>
        result.status === 'fulfilled' ? [result.value] : []
      );
      const failedRegistration = registrations.find(
        (result): result is PromiseRejectedResult => result.status === 'rejected'
      );

      if (disposed || failedRegistration) {
        unlisteners.forEach(unlisten => unlisten());
        if (!disposed && failedRegistration) {
          const message = failedRegistration.reason instanceof Error
            ? failedRegistration.reason.message
            : typeof failedRegistration.reason === 'string'
              ? failedRegistration.reason
              : 'Failed to listen for FluidAudio model updates';
          setListenersReady(false);
          setError(message);
          setLoading(false);
        }
        return;
      }

      registeredUnlisteners = unlisteners;
      setListenersReady(true);
    };

    setupListeners();

    return () => {
      disposed = true;
      registeredUnlisteners.forEach(unlisten => unlisten());
    };
  }, [getDisplayName, clearLocalDownloadingState, saveModelSelection, listenerRetryCount]);

  // Fetch models once listeners are ready
  useEffect(() => {
    if (initialized || !listenersReady) return;
    fetchModels();
  }, [initialized, listenersReady, fetchModels]);

  const handleRetry = useCallback(() => {
    if (!listenersReady) {
      setLoading(true);
      setError(null);
      setListenerRetryCount(prev => prev + 1);
    } else {
      void fetchModels();
    }
  }, [listenersReady, fetchModels]);

  const downloadModel = async (modelName: string) => {
    if (downloadingModels.has(modelName)) return;

    const displayName = getDisplayName(modelName);
    const initialStatus: ModelStatus = { Downloading: { progress: 0 } };
    latestStatusByModelRef.current.set(modelName, initialStatus);
    latestStageByModelRef.current.set(modelName, 'download');

    setStageByModel(prev => ({ ...prev, [modelName]: 'download' }));
    setDownloadingModels(prev => new Set([...prev, modelName]));
    setModels(prevModels =>
      prevModels.map(model =>
        model.name === modelName
          ? { ...model, status: initialStatus }
          : model
      )
    );

    toast.info(`Downloading ${displayName}...`, {
      description: 'Downloading and compiling for the Apple Neural Engine',
      duration: 5000
    });

    try {
      await invoke('fluid_download_model', { modelName });
      clearLocalDownloadingState(modelName);
      latestStatusByModelRef.current.delete(modelName);

      try {
        const modelList = await invoke<FluidModelInfo[]>('fluid_get_available_models');
        setModels(
          modelList.map(model => ({
            ...model,
            status: latestStatusByModelRef.current.get(model.name) ?? model.status
          }))
        );
      } catch (refreshErr) {
        console.error('Failed to refresh FluidAudio models after download completion:', refreshErr);
        setModels(prev =>
          prev.map(model =>
            model.name === modelName ? { ...model, status: 'Available' } : model
          )
        );
      }
    } catch (err) {
      console.error('FluidAudio download failed:', err);
      const errorMessage = err instanceof Error ? err.message : typeof err === 'string' ? err : 'Download failed';
      const errorStatus: ModelStatus = { Error: errorMessage };
      clearLocalDownloadingState(modelName);
      latestStatusByModelRef.current.set(modelName, errorStatus);

      setModels(prev =>
        prev.map(model =>
          model.name === modelName ? { ...model, status: errorStatus } : model
        )
      );

      try {
        const modelList = await invoke<FluidModelInfo[]>('fluid_get_available_models');
        setModels(
          modelList.map(model => {
            if (model.name === modelName && model.status === 'Missing') {
              return { ...model, status: errorStatus };
            }
            return {
              ...model,
              status: latestStatusByModelRef.current.get(model.name) ?? model.status
            };
          })
        );
      } catch (refreshErr) {
        console.error('Failed to refresh FluidAudio models after download error:', refreshErr);
      }
    }
  };

  const selectModel = async (modelName: string) => {
    if (onModelSelect) {
      onModelSelect(modelName);
    }

    if (autoSave) {
      await saveModelSelection(modelName);
    }

    const displayName = getDisplayName(modelName);
    toast.success(`Switched to ${displayName}`, {
      duration: 3000
    });
  };

  const deleteModel = async (modelName: string) => {
    if (deletingModels.has(modelName)) return;
    const displayName = getDisplayName(modelName);

    setDeletingModels(prev => new Set([...prev, modelName]));
    try {
      await invoke('fluid_delete_model', { modelName });
      latestStatusByModelRef.current.set(modelName, 'Missing');

      const modelList = await invoke<FluidModelInfo[]>('fluid_get_available_models');
      setModels(modelList);

      toast.success(`${displayName} deleted`, {
        description: 'Model removed to free up space',
        duration: 3000
      });
    } catch (err) {
      console.error('Failed to delete FluidAudio model:', err);
      const errorMessage = err instanceof Error ? err.message : typeof err === 'string' ? err : 'Delete failed';
      toast.error(`Failed to delete ${displayName}`, {
        description: errorMessage,
        duration: 4000
      });
    } finally {
      setDeletingModels(prev => {
        const next = new Set(prev);
        next.delete(modelName);
        return next;
      });
    }
  };

  if (loading) {
    return (
      <div className={`space-y-3 ${className}`}>
        <div className="animate-pulse space-y-3">
          <div className="h-20 bg-gray-100 rounded-lg"></div>
          <div className="h-20 bg-gray-100 rounded-lg"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-4 flex items-center justify-between ${className}`}>
        <div>
          <p className="text-sm text-red-800">Failed to load models</p>
          <p className="text-xs text-red-600 mt-1">{error}</p>
        </div>
        <button
          onClick={handleRetry}
          className="bg-red-600 text-white px-3 py-1.5 rounded-md text-sm font-medium hover:bg-red-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  const canSelect = Boolean(onModelSelect);

  return (
    <div className={`space-y-3 ${className}`}>
      {models.map(model => (
        <FluidModelCard
          key={model.name}
          model={model}
          isSelected={canSelect && selectedModel === model.name}
          canSelect={canSelect}
          isRecommended={model.name === 'fluid-parakeet-v3'}
          downloadStage={stageByModel[model.name] ?? latestStageByModelRef.current.get(model.name) ?? 'download'}
          isDeleting={deletingModels.has(model.name)}
          onSelect={() => {
            if (canSelect && model.status === 'Available') {
              void selectModel(model.name);
            }
          }}
          onDownload={() => downloadModel(model.name)}
          onDelete={() => {
            if (!deletingModels.has(model.name)) {
              setModelPendingDelete(model);
            }
          }}
        />
      ))}

      {canSelect && selectedModel && (
        <motion.div
          initial={{ opacity: 0, y: -5 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-xs text-gray-500 text-center pt-2"
        >
          Using {getDisplayName(selectedModel)} for transcription
        </motion.div>
      )}

      <ConfirmationModal
        isOpen={modelPendingDelete !== null}
        text={
          modelPendingDelete
            ? `Delete ${modelPendingDelete.display_name}? You can download it again later (about ${modelPendingDelete.size_mb} MB).`
            : ''
        }
        onConfirm={() => {
          const target = modelPendingDelete;
          setModelPendingDelete(null);
          if (target) {
            void deleteModel(target.name);
          }
        }}
        onCancel={() => setModelPendingDelete(null)}
      />
    </div>
  );
}

interface FluidModelCardProps {
  model: FluidModelInfo;
  isSelected: boolean;
  canSelect: boolean;
  isRecommended: boolean;
  downloadStage: FluidLoadStage;
  isDeleting: boolean;
  onSelect: () => void;
  onDownload: () => void;
  onDelete: () => void;
}

function FluidModelCard({
  model,
  isSelected,
  canSelect,
  isRecommended,
  downloadStage,
  isDeleting,
  onSelect,
  onDownload,
  onDelete
}: FluidModelCardProps) {
  const isAvailable = model.status === 'Available';
  const isMissing = model.status === 'Missing';
  const isError = typeof model.status === 'object' && 'Error' in model.status;
  const errorMessage =
    typeof model.status === 'object' && 'Error' in model.status
      ? model.status.Error
      : null;
  const downloadProgress =
    typeof model.status === 'object' && 'Downloading' in model.status
      ? model.status.Downloading.progress
      : null;
  const displayedProgress = downloadProgress ?? 0;

  const progressLabel =
    downloadStage === 'compile'
      ? `Preparing for the Neural Engine… ${Math.round(displayedProgress)}%`
      : `Downloading… ${Math.round(displayedProgress)}%`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`
        relative rounded-lg border-2 transition-all
        ${isSelected && isAvailable
          ? 'border-blue-500 bg-blue-50'
          : isAvailable
            ? `border-gray-200 bg-white ${canSelect ? 'hover:border-gray-300' : ''}`
            : 'border-gray-200 bg-gray-50'
        }
        ${canSelect && isAvailable ? 'cursor-pointer' : 'cursor-default'}
      `}
      onClick={() => {
        if (canSelect && isAvailable) {
          onSelect();
        }
      }}
    >
      {isRecommended && (
        <div className="absolute -top-2 -right-2 bg-blue-600 text-white text-xs px-2 py-0.5 rounded-full font-medium">
          Recommended
        </div>
      )}

      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-2xl">🍎</span>
              <h3 className="font-semibold text-gray-900">{model.display_name}</h3>
              <span className="text-xs text-gray-500 font-medium bg-gray-100 px-2 py-0.5 rounded">
                {formatFileSize(model.size_mb)}
              </span>
              {isSelected && isAvailable && (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="bg-blue-600 text-white px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1"
                >
                  ✓
                </motion.span>
              )}
            </div>

            <p className="text-sm text-gray-600 ml-9">{model.description}</p>

            {isError && errorMessage && (
              <p className="text-xs text-red-600 ml-9 mt-1">{errorMessage}</p>
            )}
          </div>

          <div className="ml-4 flex items-center gap-2">
            {isAvailable && (
              <>
                <div className="flex items-center gap-1.5 text-green-600">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-xs font-medium">Ready</span>
                </div>
                {canSelect && !isSelected && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect();
                    }}
                    className="text-xs font-medium text-blue-600 hover:text-blue-700 bg-blue-50 hover:bg-blue-100 px-2.5 py-1 rounded-md transition-colors"
                  >
                    Use for transcription
                  </button>
                )}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete();
                  }}
                  disabled={isDeleting}
                  className="text-gray-400 hover:text-red-600 transition-colors p-1 disabled:opacity-50"
                  title="Delete model to free up space"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </>
            )}

            {isMissing && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDownload();
                }}
                className="bg-blue-600 text-white px-3 py-1.5 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors"
              >
                Download
              </button>
            )}

            {downloadProgress === null && isError && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDownload();
                }}
                className="bg-red-600 text-white px-3 py-1.5 rounded-md text-sm font-medium hover:bg-red-700 transition-colors"
              >
                Retry
              </button>
            )}
          </div>
        </div>

        {downloadProgress !== null && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3 pt-3 border-t border-gray-200"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-blue-600">
                {progressLabel}
              </span>
            </div>
            <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${displayedProgress}%` }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
              />
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
