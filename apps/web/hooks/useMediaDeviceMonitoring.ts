'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Severity } from '@interviehire/shared';

export type MediaDeviceCategory =
  | 'integrated_camera'
  | 'external_camera'
  | 'virtual_camera'
  | 'capture_device'
  | 'built_in_microphone'
  | 'external_microphone'
  | 'headset'
  | 'virtual_audio'
  | 'speaker'
  | 'unknown_camera'
  | 'unknown_microphone'
  | 'unknown_audio_output';

export type DetectedMediaDevice = {
  deviceId: string;
  groupId: string;
  kind: MediaDeviceKind;
  label: string;
  category: MediaDeviceCategory;
  risk: 'normal' | 'context' | 'high';
  reason: string;
};

export type MediaDeviceSnapshot = {
  capturedAt: number;
  cameras: DetectedMediaDevice[];
  microphones: DetectedMediaDevice[];
  audioOutputs: DetectedMediaDevice[];
};

type EmitProctoringEvent = (
  eventType: string,
  severity: Severity,
  metadata?: Record<string, unknown>,
) => void;

type UseMediaDeviceMonitoringOptions = {
  enabled: boolean;
  emit: EmitProctoringEvent;
  requestMicrophonePermission?: boolean;
};

const DEVICE_CHANGE_DEBOUNCE_MS = 750;
const EMPTY_SNAPSHOT: MediaDeviceSnapshot = {
  capturedAt: 0,
  cameras: [],
  microphones: [],
  audioOutputs: [],
};

const VIRTUAL_CAMERA_HINTS = [
  'virtual',
  'obs camera',
  'obs virtual',
  'snap camera',
  'manycam',
  'xsplit',
  'droidcam',
  'iriun',
  'camo',
  'epoccam',
];
const CAPTURE_DEVICE_HINTS = [
  'capture',
  'cam link',
  'elgato',
  'hdmi',
  'decklink',
];
const EXTERNAL_CAMERA_HINTS = [
  'usb',
  'webcam',
  'logitech',
  'brio',
  'lifecam',
  'streamcam',
  'razer kiyo',
];
const INTEGRATED_CAMERA_HINTS = [
  'integrated',
  'built-in',
  'facetime',
  'front camera',
];
const VIRTUAL_AUDIO_HINTS = [
  'virtual',
  'voicemeeter',
  'vb-audio',
  'blackhole',
  'soundflower',
  'loopback',
  'cable input',
  'cable output',
  'stereo mix',
  'what u hear',
];
const HEADSET_HINTS = [
  'headset',
  'headphone',
  'earphone',
  'airpods',
  'earbuds',
  'buds',
  'bluetooth',
  'hands-free',
];
const EXTERNAL_MICROPHONE_HINTS = [
  'usb',
  'wireless',
  'rode',
  'shure',
  'blue yeti',
  'hyperx',
  'elgato wave',
];
const BUILT_IN_AUDIO_HINTS = [
  'integrated',
  'built-in',
  'internal',
  'array',
  'realtek',
];

function labelHasHint(label: string, hints: string[]) {
  const normalized = label.toLowerCase();
  return hints.some((hint) => normalized.includes(hint));
}

function classifyMediaDevice(device: MediaDeviceInfo, index: number): DetectedMediaDevice {
  const label = device.label.trim() || `${device.kind} ${index + 1}`;
  const baseDevice = {
    deviceId: device.deviceId,
    groupId: device.groupId,
    kind: device.kind,
  };

  if (device.kind === 'videoinput') {
    if (labelHasHint(label, VIRTUAL_CAMERA_HINTS)) {
      return {
        ...baseDevice,
        label,
        category: 'virtual_camera',
        risk: 'high',
        reason: 'Virtual camera software detected',
      };
    }
    if (labelHasHint(label, CAPTURE_DEVICE_HINTS)) {
      return {
        ...baseDevice,
        label,
        category: 'capture_device',
        risk: 'high',
        reason: 'Video capture or HDMI capture device detected',
      };
    }
    if (labelHasHint(label, EXTERNAL_CAMERA_HINTS)) {
      return {
        ...baseDevice,
        label,
        category: 'external_camera',
        risk: 'context',
        reason: 'Likely external webcam',
      };
    }
    if (labelHasHint(label, INTEGRATED_CAMERA_HINTS)) {
      return {
        ...baseDevice,
        label,
        category: 'integrated_camera',
        risk: 'normal',
        reason: 'Likely integrated camera',
      };
    }
    return {
      ...baseDevice,
      label,
      category: 'unknown_camera',
      risk: 'normal',
      reason: 'Camera type could not be determined from its label',
    };
  }

  if (labelHasHint(label, VIRTUAL_AUDIO_HINTS)) {
    return {
      ...baseDevice,
      label,
      category: 'virtual_audio',
      risk: 'high',
      reason: 'Virtual or system-audio routing device detected',
    };
  }
  if (labelHasHint(label, HEADSET_HINTS)) {
    return {
      ...baseDevice,
      label,
      category: 'headset',
      risk: 'context',
      reason: 'Likely headset or Bluetooth audio device',
    };
  }
  if (device.kind === 'audioinput') {
    if (labelHasHint(label, EXTERNAL_MICROPHONE_HINTS)) {
      return {
        ...baseDevice,
        label,
        category: 'external_microphone',
        risk: 'context',
        reason: 'Likely external microphone',
      };
    }
    if (labelHasHint(label, BUILT_IN_AUDIO_HINTS)) {
      return {
        ...baseDevice,
        label,
        category: 'built_in_microphone',
        risk: 'normal',
        reason: 'Likely built-in microphone',
      };
    }
    return {
      ...baseDevice,
      label,
      category: 'unknown_microphone',
      risk: 'normal',
      reason: 'Microphone type could not be determined from its label',
    };
  }

  return {
    ...baseDevice,
    label,
    category: device.kind === 'audiooutput' ? 'speaker' : 'unknown_audio_output',
    risk: 'normal',
    reason: 'Audio output device',
  };
}

function createSnapshot(devices: MediaDeviceInfo[]): MediaDeviceSnapshot {
  const concreteKinds = new Set(
    devices
      .filter((device) => device.deviceId !== 'default' && device.deviceId !== 'communications')
      .map((device) => device.kind),
  );
  const classified = devices
    .filter((device) => {
      const isBrowserAlias = device.deviceId === 'default' || device.deviceId === 'communications';
      return !isBrowserAlias || !concreteKinds.has(device.kind);
    })
    .map(classifyMediaDevice);

  return {
    capturedAt: Date.now(),
    cameras: classified.filter((device) => device.kind === 'videoinput'),
    microphones: classified.filter((device) => device.kind === 'audioinput'),
    audioOutputs: classified.filter((device) => device.kind === 'audiooutput'),
  };
}

function deviceKey(device: DetectedMediaDevice) {
  return `${device.kind}:${device.deviceId || device.groupId || device.label}`;
}

function publicDeviceMetadata(device: DetectedMediaDevice) {
  return {
    kind: device.kind,
    label: device.label,
    category: device.category,
    risk: device.risk,
    reason: device.reason,
  };
}

function snapshotCounts(snapshot: MediaDeviceSnapshot) {
  return {
    cameras: snapshot.cameras.length,
    microphones: snapshot.microphones.length,
    audioOutputs: snapshot.audioOutputs.length,
  };
}

function publicSnapshotMetadata(snapshot: MediaDeviceSnapshot) {
  return {
    capturedAt: snapshot.capturedAt,
    counts: snapshotCounts(snapshot),
    devices: [
      ...snapshot.cameras,
      ...snapshot.microphones,
      ...snapshot.audioOutputs,
    ].map(publicDeviceMetadata),
  };
}

export function useMediaDeviceMonitoring({
  enabled,
  emit,
  requestMicrophonePermission = true,
}: UseMediaDeviceMonitoringOptions) {
  const [devices, setDevices] = useState<MediaDeviceSnapshot>(EMPTY_SNAPSHOT);
  const [microphonePermission, setMicrophonePermission] = useState<
    'unknown' | 'requesting' | 'granted' | 'denied'
  >('unknown');
  const snapshotRef = useRef<MediaDeviceSnapshot | null>(null);
  const scanInFlightRef = useRef(false);
  const permissionRequestedRef = useRef(false);
  const deviceChangeTimerRef = useRef<number | null>(null);

  const scanDevices = useCallback(async (
    trigger: 'initial' | 'devicechange' | 'manual' = 'manual',
  ) => {
    if (
      typeof navigator === 'undefined'
      || !navigator.mediaDevices?.enumerateDevices
      || scanInFlightRef.current
    ) {
      return null;
    }

    scanInFlightRef.current = true;

    try {
      const snapshot = createSnapshot(await navigator.mediaDevices.enumerateDevices());
      const previous = snapshotRef.current;
      snapshotRef.current = snapshot;
      setDevices(snapshot);

      if (!previous || trigger === 'initial') {
        emit('MEDIA_DEVICE_INVENTORY_RECORDED', 'LOW', {
          trigger,
          ...publicSnapshotMetadata(snapshot),
        });
      }

      if (previous) {
        const previousByKey = new Map(
          [...previous.cameras, ...previous.microphones, ...previous.audioOutputs]
            .map((device) => [deviceKey(device), device]),
        );
        const currentByKey = new Map(
          [...snapshot.cameras, ...snapshot.microphones, ...snapshot.audioOutputs]
            .map((device) => [deviceKey(device), device]),
        );

        for (const [key, device] of currentByKey) {
          if (previousByKey.has(key)) continue;
          emit(
            'MEDIA_DEVICE_CONNECTED',
            device.risk === 'high' ? 'HIGH' : device.risk === 'context' ? 'MEDIUM' : 'LOW',
            {
              trigger,
              device: publicDeviceMetadata(device),
              counts: snapshotCounts(snapshot),
            },
          );
        }

        for (const [key, device] of previousByKey) {
          if (currentByKey.has(key)) continue;
          emit('MEDIA_DEVICE_DISCONNECTED', 'LOW', {
            trigger,
            device: publicDeviceMetadata(device),
            counts: snapshotCounts(snapshot),
          });
        }
      }

      const highRiskDevices = [
        ...snapshot.cameras,
        ...snapshot.microphones,
        ...snapshot.audioOutputs,
      ].filter((device) => device.risk === 'high');

      if (highRiskDevices.length > 0 && (!previous || trigger !== 'manual')) {
        emit('SUSPICIOUS_MEDIA_DEVICE_DETECTED', 'HIGH', {
          trigger,
          devices: highRiskDevices.map(publicDeviceMetadata),
        });
      }

      const previousHadMultipleCameras = (previous?.cameras.length ?? 0) > 1;
      const previousHadMultipleMicrophones = (previous?.microphones.length ?? 0) > 1;
      const multipleInputsAppeared =
        (snapshot.cameras.length > 1 && !previousHadMultipleCameras)
        || (snapshot.microphones.length > 1 && !previousHadMultipleMicrophones);
      const initialMultipleInputs =
        trigger === 'initial'
        && (snapshot.cameras.length > 1 || snapshot.microphones.length > 1);

      if (multipleInputsAppeared || initialMultipleInputs) {
        emit('MULTIPLE_MEDIA_INPUTS_DETECTED', 'MEDIUM', {
          trigger,
          cameraCount: snapshot.cameras.length,
          microphoneCount: snapshot.microphones.length,
          cameras: snapshot.cameras.map(publicDeviceMetadata),
          microphones: snapshot.microphones.map(publicDeviceMetadata),
        });
      }

      return snapshot;
    } catch (error) {
      emit('MEDIA_DEVICE_SCAN_FAILED', 'MEDIUM', {
        trigger,
        message: error instanceof Error ? error.message : 'Unable to enumerate media devices',
      });
      return null;
    } finally {
      scanInFlightRef.current = false;
    }
  }, [emit]);

  useEffect(() => {
    if (!enabled) return;

    const prepareDeviceInventory = async () => {
      if (
        !permissionRequestedRef.current
        && requestMicrophonePermission
        && typeof navigator !== 'undefined'
        && navigator.mediaDevices?.getUserMedia
      ) {
        permissionRequestedRef.current = true;
        setMicrophonePermission('requesting');
        const requestedAt = Date.now();

        try {
          const permissionStream = await navigator.mediaDevices.getUserMedia({
            video: false,
            audio: true,
          });
          permissionStream.getTracks().forEach((track) => track.stop());
          setMicrophonePermission('granted');
          emit('MICROPHONE_PERMISSION_GRANTED', 'LOW', {
            requestedAt,
            grantedAt: Date.now(),
            purpose: 'media_device_inventory',
          });
        } catch (error) {
          setMicrophonePermission('denied');
          emit('MICROPHONE_PERMISSION_DENIED', 'MEDIUM', {
            requestedAt,
            message: error instanceof Error ? error.message : 'Microphone permission denied',
            purpose: 'media_device_inventory',
          });
        }
      }

      await scanDevices('initial');
    };

    void prepareDeviceInventory();
  }, [emit, enabled, requestMicrophonePermission, scanDevices]);

  useEffect(() => {
    if (
      !enabled
      || typeof window === 'undefined'
      || !navigator.mediaDevices?.addEventListener
    ) {
      return;
    }

    const handleDeviceChange = () => {
      if (deviceChangeTimerRef.current !== null) {
        window.clearTimeout(deviceChangeTimerRef.current);
      }
      deviceChangeTimerRef.current = window.setTimeout(() => {
        deviceChangeTimerRef.current = null;
        void scanDevices('devicechange');
      }, DEVICE_CHANGE_DEBOUNCE_MS);
    };

    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);

    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
      if (deviceChangeTimerRef.current !== null) {
        window.clearTimeout(deviceChangeTimerRef.current);
        deviceChangeTimerRef.current = null;
      }
    };
  }, [enabled, scanDevices]);

  return {
    devices,
    microphonePermission,
    scanDevices,
  };
}
