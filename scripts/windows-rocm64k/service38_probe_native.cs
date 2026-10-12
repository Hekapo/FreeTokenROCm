using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class Service38Native {
    [StructLayout(LayoutKind.Sequential)]
    public class MemoryStatus {
        public uint Length = 64, Load;
        public ulong TotalPhysical, AvailablePhysical, TotalPageFile, AvailablePageFile;
        public ulong TotalVirtual, AvailableVirtual, AvailableExtendedVirtual;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GlobalMemoryStatusEx([In, Out] MemoryStatus memory);

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimit {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters { public ulong ReadOps, WriteOps, OtherOps, ReadBytes, WriteBytes, OtherBytes; }
    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimit {
        public BasicLimit Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    public sealed class Job : SafeHandleZeroOrMinusOneIsInvalid {
        public Job() : base(true) { }
        protected override bool ReleaseHandle() { return CloseHandle(handle); }
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern Job CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(Job job, int kind, ref ExtendedLimit info, uint length);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(Job job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);
    public static Job CreateProbeJob() {
        var job = CreateJobObject(IntPtr.Zero, null);
        if (job.IsInvalid) {
            int error = Marshal.GetLastWin32Error(); job.Dispose(); throw new Win32Exception(error);
        }
        var limits = new ExtendedLimit();
        limits.Basic.Flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (!SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf<ExtendedLimit>())) {
            int error = Marshal.GetLastWin32Error(); job.Dispose(); throw new Win32Exception(error);
        }
        return job;
    }
    public static void Attach(Job job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process)) throw new Win32Exception(Marshal.GetLastWin32Error());
    }
}
