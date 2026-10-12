using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

// DXGI enumeration only: no compute device, HIP runtime or GPU workload.
public static class Service38GpuIdentity {
    [StructLayout(LayoutKind.Sequential)]
    private struct Luid { public uint Low; public int High; }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct Desc {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string Description;
        public uint VendorId, DeviceId, SubsystemId, Revision;
        public UIntPtr DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory;
        public Luid AdapterLuid;
        public uint Flags;
    }
    public sealed class Adapter {
        public string description, counter_instance;
        public uint vendor_id, device_id, flags;
        public ulong dedicated_video_bytes;
    }
    [DllImport("dxgi.dll", ExactSpelling = true)]
    private static extern int CreateDXGIFactory1(ref Guid iid, out IntPtr factory);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int EnumAdapters1(IntPtr factory, uint index, out IntPtr adapter);
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int GetDesc1(IntPtr adapter, out Desc desc);
    private static T Method<T>(IntPtr instance, int slot) where T : Delegate {
        IntPtr table = Marshal.ReadIntPtr(instance);
        return Marshal.GetDelegateForFunctionPointer<T>(Marshal.ReadIntPtr(table, slot * IntPtr.Size));
    }
    public static Adapter[] Read() {
        Guid iid = new Guid("770aae78-f26f-4dba-a829-253c83d1b387");
        Marshal.ThrowExceptionForHR(CreateDXGIFactory1(ref iid, out IntPtr factory));
        try {
            // IUnknown(3), IDXGIObject(4), IDXGIFactory(5), EnumAdapters1.
            var enumerate = Method<EnumAdapters1>(factory, 12);
            var result = new List<Adapter>();
            for (uint index = 0; index < 32; index++) {
                int status = enumerate(factory, index, out IntPtr adapter);
                if (status == unchecked((int)0x887A0002)) return result.ToArray();
                Marshal.ThrowExceptionForHR(status);
                try {
                    // IUnknown(3), IDXGIObject(4), IDXGIAdapter(3), GetDesc1.
                    Marshal.ThrowExceptionForHR(Method<GetDesc1>(adapter, 10)(adapter, out Desc desc));
                    result.Add(new Adapter {
                        description = desc.Description, vendor_id = desc.VendorId, device_id = desc.DeviceId,
                        flags = desc.Flags, dedicated_video_bytes = desc.DedicatedVideoMemory.ToUInt64(),
                        counter_instance = "luid_0x" + unchecked((uint)desc.AdapterLuid.High).ToString("X8") +
                            "_0x" + desc.AdapterLuid.Low.ToString("X8") + "_phys_0"
                    });
                } finally { Marshal.Release(adapter); }
            }
            throw new InvalidOperationException("Unexpected DXGI adapter count exceeds 32");
        } finally { Marshal.Release(factory); }
    }
}
