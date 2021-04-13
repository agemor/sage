#[allow(unused_imports)]
use std::sync::Arc;
#[allow(unused_imports)]
use std::vec::IntoIter as VecIntoIter;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor::DescriptorBufferDesc;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor::DescriptorDesc;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor::DescriptorDescTy;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor::DescriptorImageDesc;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor::DescriptorImageDescArray;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor::DescriptorImageDescDimensions;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor::ShaderStages;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor_set::DescriptorSet;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor_set::UnsafeDescriptorSet;
#[allow(unused_imports)]
use vulkano::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
#[allow(unused_imports)]
use vulkano::descriptor::pipeline_layout::PipelineLayout;
#[allow(unused_imports)]
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
#[allow(unused_imports)]
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
#[allow(unused_imports)]
use vulkano::device::Device;
#[allow(unused_imports)]
use vulkano::pipeline::shader::SpecializationConstants as SpecConstsTrait;
#[allow(unused_imports)]
use vulkano::pipeline::shader::SpecializationMapEntry;
pub struct Shader {
    shader: ::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule>,
}
impl Shader {
    #[doc = r" Loads the shader in Vulkan as a `ShaderModule`."]
    #[inline]
    #[allow(unsafe_code)]
    pub fn load(
        device: ::std::sync::Arc<::vulkano::device::Device>,
    ) -> Result<Shader, ::vulkano::OomError> {
        if !device.loaded_extensions().khr_storage_buffer_storage_class {
            panic!(
                "Device extension {:?} required",
                "khr_storage_buffer_storage_class"
            );
        }
        if !device.loaded_extensions().khr_storage_buffer_storage_class {
            panic!(
                "Device extension {:?} required",
                "khr_storage_buffer_storage_class"
            );
        }
        if !device.loaded_extensions().khr_storage_buffer_storage_class {
            panic!(
                "Device extension {:?} required",
                "khr_storage_buffer_storage_class"
            );
        }
        if !device.loaded_extensions().khr_storage_buffer_storage_class {
            panic!(
                "Device extension {:?} required",
                "khr_storage_buffer_storage_class"
            );
        }
        if !device.loaded_extensions().khr_storage_buffer_storage_class {
            panic!(
                "Device extension {:?} required",
                "khr_storage_buffer_storage_class"
            );
        }
        if !device.loaded_extensions().khr_storage_buffer_storage_class {
            panic!(
                "Device extension {:?} required",
                "khr_storage_buffer_storage_class"
            );
        }
        if !device.loaded_extensions().khr_storage_buffer_storage_class {
            panic!(
                "Device extension {:?} required",
                "khr_storage_buffer_storage_class"
            );
        }
        static WORDS: &[u32] = &[
            119734787u32,
            66304u32,
            851978u32,
            119u32,
            0u32,
            131089u32,
            1u32,
            393227u32,
            1u32,
            1280527431u32,
            1685353262u32,
            808793134u32,
            0u32,
            196622u32,
            0u32,
            1u32,
            393231u32,
            5u32,
            4u32,
            1852399981u32,
            0u32,
            24u32,
            393232u32,
            4u32,
            17u32,
            1u32,
            1u32,
            1u32,
            196611u32,
            2u32,
            450u32,
            655364u32,
            1197427783u32,
            1279741775u32,
            1885560645u32,
            1953718128u32,
            1600482425u32,
            1701734764u32,
            1919509599u32,
            1769235301u32,
            25974u32,
            524292u32,
            1197427783u32,
            1279741775u32,
            1852399429u32,
            1685417059u32,
            1768185701u32,
            1952671090u32,
            6649449u32,
            262149u32,
            4u32,
            1852399981u32,
            0u32,
            327685u32,
            8u32,
            1701080681u32,
            1970233208u32,
            116u32,
            393221u32,
            13u32,
            1752397136u32,
            1936617283u32,
            1953390964u32,
            115u32,
            327686u32,
            13u32,
            0u32,
            1601009006u32,
            7170404u32,
            327686u32,
            13u32,
            1u32,
            1936090735u32,
            7566437u32,
            327686u32,
            13u32,
            2u32,
            1769108595u32,
            7562596u32,
            196613u32,
            15u32,
            0u32,
            524293u32,
            24u32,
            1197436007u32,
            1633841004u32,
            1986939244u32,
            1952539503u32,
            1231974249u32,
            68u32,
            327685u32,
            30u32,
            1701080681u32,
            1852399480u32,
            48u32,
            327685u32,
            33u32,
            1701080681u32,
            1852399480u32,
            49u32,
            196613u32,
            37u32,
            7169394u32,
            196613u32,
            39u32,
            105u32,
            327685u32,
            50u32,
            1769108595u32,
            1601398116u32,
            7632239u32,
            327685u32,
            56u32,
            1769108595u32,
            1601398116u32,
            3173993u32,
            327685u32,
            63u32,
            1769108595u32,
            1601398116u32,
            3239529u32,
            196613u32,
            70u32,
            99u32,
            196613u32,
            93u32,
            12408u32,
            327685u32,
            95u32,
            1717990722u32,
            1850307173u32,
            48u32,
            327686u32,
            95u32,
            0u32,
            1635017060u32,
            0u32,
            327685u32,
            97u32,
            1717990754u32,
            1767862885u32,
            12398u32,
            196613u32,
            102u32,
            12664u32,
            327685u32,
            104u32,
            1717990722u32,
            1850307173u32,
            49u32,
            327686u32,
            104u32,
            0u32,
            1635017060u32,
            0u32,
            327685u32,
            106u32,
            1717990754u32,
            1767862885u32,
            12654u32,
            196613u32,
            110u32,
            121u32,
            327685u32,
            113u32,
            1717990722u32,
            1968140901u32,
            116u32,
            327686u32,
            113u32,
            0u32,
            1635017060u32,
            0u32,
            327685u32,
            115u32,
            1717990754u32,
            1868526181u32,
            29813u32,
            262215u32,
            10u32,
            6u32,
            4u32,
            262215u32,
            12u32,
            6u32,
            4u32,
            327752u32,
            13u32,
            0u32,
            35u32,
            0u32,
            327752u32,
            13u32,
            1u32,
            35u32,
            4u32,
            327752u32,
            13u32,
            2u32,
            35u32,
            16u32,
            196679u32,
            13u32,
            2u32,
            262215u32,
            24u32,
            11u32,
            28u32,
            262215u32,
            94u32,
            6u32,
            4u32,
            262216u32,
            95u32,
            0u32,
            24u32,
            327752u32,
            95u32,
            0u32,
            35u32,
            0u32,
            196679u32,
            95u32,
            2u32,
            262215u32,
            97u32,
            34u32,
            0u32,
            262215u32,
            97u32,
            33u32,
            1u32,
            262215u32,
            103u32,
            6u32,
            4u32,
            262216u32,
            104u32,
            0u32,
            24u32,
            327752u32,
            104u32,
            0u32,
            35u32,
            0u32,
            196679u32,
            104u32,
            2u32,
            262215u32,
            106u32,
            34u32,
            0u32,
            262215u32,
            106u32,
            33u32,
            2u32,
            262215u32,
            112u32,
            6u32,
            4u32,
            262216u32,
            113u32,
            0u32,
            25u32,
            327752u32,
            113u32,
            0u32,
            35u32,
            0u32,
            196679u32,
            113u32,
            2u32,
            262215u32,
            115u32,
            34u32,
            0u32,
            262215u32,
            115u32,
            33u32,
            0u32,
            131091u32,
            2u32,
            196641u32,
            3u32,
            2u32,
            262165u32,
            6u32,
            32u32,
            0u32,
            262176u32,
            7u32,
            7u32,
            6u32,
            262187u32,
            6u32,
            9u32,
            3u32,
            262172u32,
            10u32,
            6u32,
            9u32,
            262187u32,
            6u32,
            11u32,
            24u32,
            262172u32,
            12u32,
            6u32,
            11u32,
            327710u32,
            13u32,
            6u32,
            10u32,
            12u32,
            262176u32,
            14u32,
            9u32,
            13u32,
            262203u32,
            14u32,
            15u32,
            9u32,
            262165u32,
            16u32,
            32u32,
            1u32,
            262187u32,
            16u32,
            17u32,
            1u32,
            262187u32,
            16u32,
            18u32,
            0u32,
            262176u32,
            19u32,
            9u32,
            6u32,
            262167u32,
            22u32,
            6u32,
            3u32,
            262176u32,
            23u32,
            1u32,
            22u32,
            262203u32,
            23u32,
            24u32,
            1u32,
            262187u32,
            6u32,
            25u32,
            0u32,
            262176u32,
            26u32,
            1u32,
            6u32,
            262187u32,
            16u32,
            34u32,
            2u32,
            131092u32,
            48u32,
            262187u32,
            6u32,
            59u32,
            1u32,
            262187u32,
            6u32,
            66u32,
            2u32,
            196630u32,
            91u32,
            32u32,
            262176u32,
            92u32,
            7u32,
            91u32,
            196637u32,
            94u32,
            91u32,
            196638u32,
            95u32,
            94u32,
            262176u32,
            96u32,
            12u32,
            95u32,
            262203u32,
            96u32,
            97u32,
            12u32,
            262176u32,
            99u32,
            12u32,
            91u32,
            196637u32,
            103u32,
            91u32,
            196638u32,
            104u32,
            103u32,
            262176u32,
            105u32,
            12u32,
            104u32,
            262203u32,
            105u32,
            106u32,
            12u32,
            262187u32,
            91u32,
            111u32,
            0u32,
            196637u32,
            112u32,
            91u32,
            196638u32,
            113u32,
            112u32,
            262176u32,
            114u32,
            12u32,
            113u32,
            262203u32,
            114u32,
            115u32,
            12u32,
            327734u32,
            2u32,
            4u32,
            0u32,
            3u32,
            131320u32,
            5u32,
            262203u32,
            7u32,
            8u32,
            7u32,
            262203u32,
            7u32,
            30u32,
            7u32,
            262203u32,
            7u32,
            33u32,
            7u32,
            262203u32,
            7u32,
            37u32,
            7u32,
            262203u32,
            7u32,
            39u32,
            7u32,
            262203u32,
            7u32,
            50u32,
            7u32,
            262203u32,
            7u32,
            56u32,
            7u32,
            262203u32,
            7u32,
            63u32,
            7u32,
            262203u32,
            7u32,
            70u32,
            7u32,
            262203u32,
            92u32,
            93u32,
            7u32,
            262203u32,
            92u32,
            102u32,
            7u32,
            262203u32,
            92u32,
            110u32,
            7u32,
            393281u32,
            19u32,
            20u32,
            15u32,
            17u32,
            18u32,
            262205u32,
            6u32,
            21u32,
            20u32,
            327745u32,
            26u32,
            27u32,
            24u32,
            25u32,
            262205u32,
            6u32,
            28u32,
            27u32,
            327808u32,
            6u32,
            29u32,
            21u32,
            28u32,
            196670u32,
            8u32,
            29u32,
            393281u32,
            19u32,
            31u32,
            15u32,
            17u32,
            17u32,
            262205u32,
            6u32,
            32u32,
            31u32,
            196670u32,
            30u32,
            32u32,
            393281u32,
            19u32,
            35u32,
            15u32,
            17u32,
            34u32,
            262205u32,
            6u32,
            36u32,
            35u32,
            196670u32,
            33u32,
            36u32,
            262205u32,
            6u32,
            38u32,
            8u32,
            196670u32,
            37u32,
            38u32,
            196670u32,
            39u32,
            25u32,
            131321u32,
            40u32,
            131320u32,
            40u32,
            262390u32,
            42u32,
            43u32,
            0u32,
            131321u32,
            44u32,
            131320u32,
            44u32,
            262205u32,
            6u32,
            45u32,
            39u32,
            327745u32,
            19u32,
            46u32,
            15u32,
            18u32,
            262205u32,
            6u32,
            47u32,
            46u32,
            327856u32,
            48u32,
            49u32,
            45u32,
            47u32,
            262394u32,
            49u32,
            41u32,
            42u32,
            131320u32,
            41u32,
            262205u32,
            6u32,
            51u32,
            39u32,
            327812u32,
            6u32,
            52u32,
            51u32,
            9u32,
            327808u32,
            6u32,
            53u32,
            52u32,
            25u32,
            393281u32,
            19u32,
            54u32,
            15u32,
            34u32,
            53u32,
            262205u32,
            6u32,
            55u32,
            54u32,
            196670u32,
            50u32,
            55u32,
            262205u32,
            6u32,
            57u32,
            39u32,
            327812u32,
            6u32,
            58u32,
            57u32,
            9u32,
            327808u32,
            6u32,
            60u32,
            58u32,
            59u32,
            393281u32,
            19u32,
            61u32,
            15u32,
            34u32,
            60u32,
            262205u32,
            6u32,
            62u32,
            61u32,
            196670u32,
            56u32,
            62u32,
            262205u32,
            6u32,
            64u32,
            39u32,
            327812u32,
            6u32,
            65u32,
            64u32,
            9u32,
            327808u32,
            6u32,
            67u32,
            65u32,
            66u32,
            393281u32,
            19u32,
            68u32,
            15u32,
            34u32,
            67u32,
            262205u32,
            6u32,
            69u32,
            68u32,
            196670u32,
            63u32,
            69u32,
            262205u32,
            6u32,
            71u32,
            37u32,
            262205u32,
            6u32,
            72u32,
            50u32,
            327814u32,
            6u32,
            73u32,
            71u32,
            72u32,
            196670u32,
            70u32,
            73u32,
            262205u32,
            6u32,
            74u32,
            70u32,
            262205u32,
            6u32,
            75u32,
            50u32,
            327812u32,
            6u32,
            76u32,
            74u32,
            75u32,
            262205u32,
            6u32,
            77u32,
            37u32,
            327810u32,
            6u32,
            78u32,
            77u32,
            76u32,
            196670u32,
            37u32,
            78u32,
            262205u32,
            6u32,
            79u32,
            70u32,
            262205u32,
            6u32,
            80u32,
            56u32,
            327812u32,
            6u32,
            81u32,
            79u32,
            80u32,
            262205u32,
            6u32,
            82u32,
            30u32,
            327808u32,
            6u32,
            83u32,
            82u32,
            81u32,
            196670u32,
            30u32,
            83u32,
            262205u32,
            6u32,
            84u32,
            70u32,
            262205u32,
            6u32,
            85u32,
            63u32,
            327812u32,
            6u32,
            86u32,
            84u32,
            85u32,
            262205u32,
            6u32,
            87u32,
            33u32,
            327808u32,
            6u32,
            88u32,
            87u32,
            86u32,
            196670u32,
            33u32,
            88u32,
            131321u32,
            43u32,
            131320u32,
            43u32,
            262205u32,
            6u32,
            89u32,
            39u32,
            327808u32,
            6u32,
            90u32,
            89u32,
            17u32,
            196670u32,
            39u32,
            90u32,
            131321u32,
            40u32,
            131320u32,
            42u32,
            262205u32,
            6u32,
            98u32,
            30u32,
            393281u32,
            99u32,
            100u32,
            97u32,
            18u32,
            98u32,
            262205u32,
            91u32,
            101u32,
            100u32,
            196670u32,
            93u32,
            101u32,
            262205u32,
            6u32,
            107u32,
            33u32,
            393281u32,
            99u32,
            108u32,
            106u32,
            18u32,
            107u32,
            262205u32,
            91u32,
            109u32,
            108u32,
            196670u32,
            102u32,
            109u32,
            196670u32,
            110u32,
            111u32,
            262205u32,
            6u32,
            116u32,
            8u32,
            262205u32,
            91u32,
            117u32,
            110u32,
            393281u32,
            99u32,
            118u32,
            115u32,
            18u32,
            116u32,
            196670u32,
            118u32,
            117u32,
            65789u32,
            65592u32,
        ];
        unsafe {
            Ok(Shader {
                shader: ::vulkano::pipeline::shader::ShaderModule::from_words(device, WORDS)?,
            })
        }
    }
    #[doc = r" Returns the module that was created."]
    #[allow(dead_code)]
    #[inline]
    pub fn module(&self) -> &::std::sync::Arc<::vulkano::pipeline::shader::ShaderModule> {
        &self.shader
    }
    #[doc = r" Returns a logical struct describing the entry point named `{ep_name}`."]
    #[inline]
    #[allow(unsafe_code)]
    pub fn main_entry_point(&self) -> ::vulkano::pipeline::shader::ComputeEntryPoint<(), Layout> {
        unsafe {
            #[allow(dead_code)]
            static NAME: [u8; 5usize] = [109u8, 97u8, 105u8, 110u8, 0];
            self.shader.compute_entry_point(
                ::std::ffi::CStr::from_ptr(NAME.as_ptr() as *const _),
                Layout(ShaderStages {
                    compute: true,
                    ..ShaderStages::none()
                }),
            )
        }
    }
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MainInput;
#[allow(unsafe_code)]
unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for MainInput {
    type Iter = MainInputIter;
    fn elements(&self) -> MainInputIter {
        MainInputIter { num: 0 }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct MainInputIter {
    num: u16,
}
impl Iterator for MainInputIter {
    type Item = ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = 0usize - self.num as usize;
        (len, Some(len))
    }
}
impl ExactSizeIterator for MainInputIter {}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MainOutput;
#[allow(unsafe_code)]
unsafe impl ::vulkano::pipeline::shader::ShaderInterfaceDef for MainOutput {
    type Iter = MainOutputIter;
    fn elements(&self) -> MainOutputIter {
        MainOutputIter { num: 0 }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct MainOutputIter {
    num: u16,
}
impl Iterator for MainOutputIter {
    type Item = ::vulkano::pipeline::shader::ShaderInterfaceDefEntry;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = 0usize - self.num as usize;
        (len, Some(len))
    }
}
impl ExactSizeIterator for MainOutputIter {}
pub mod ty {
    #[repr(C)]
    #[derive(Copy)]
    #[allow(non_snake_case)]
    pub struct PushConstants {
        pub num_dim: u32,
        pub offsets: [u32; 3usize],
        pub strides: [u32; 24usize],
    }
    impl Clone for PushConstants {
        fn clone(&self) -> Self {
            PushConstants {
                num_dim: self.num_dim,
                offsets: self.offsets,
                strides: self.strides,
            }
        }
    }
    #[repr(C)]
    #[allow(non_snake_case)]
    pub struct BufferIn0 {
        pub data: [f32],
    }
    #[repr(C)]
    #[allow(non_snake_case)]
    pub struct BufferIn1 {
        pub data: [f32],
    }
    #[repr(C)]
    #[allow(non_snake_case)]
    pub struct BufferOut {
        pub data: [f32],
    }
}
#[derive(Debug, Clone)]
pub struct Layout(pub ShaderStages);
#[allow(unsafe_code)]
unsafe impl PipelineLayoutDesc for Layout {
    fn num_sets(&self) -> usize {
        1usize
    }
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set {
            0usize => Some(3usize),
            _ => None,
        }
    }
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        match (set, binding) {
            (0usize, 1usize) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: None,
                    storage: true,
                }),
                array_count: 1u32,
                stages: self.0.clone(),
                readonly: true,
            }),
            (0usize, 2usize) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: None,
                    storage: true,
                }),
                array_count: 1u32,
                stages: self.0.clone(),
                readonly: true,
            }),
            (0usize, 0usize) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: None,
                    storage: true,
                }),
                array_count: 1u32,
                stages: self.0.clone(),
                readonly: true,
            }),
            _ => None,
        }
    }
    fn num_push_constants_ranges(&self) -> usize {
        1usize
    }
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        if num != 0 || 112usize == 0 {
            None
        } else {
            Some(PipelineLayoutDescPcRange {
                offset: 0,
                size: 112usize,
                stages: ShaderStages::all(),
            })
        }
    }
}
#[derive(Debug, Copy, Clone)]
#[allow(non_snake_case)]
#[repr(C)]
pub struct SpecializationConstants {}
impl Default for SpecializationConstants {
    fn default() -> SpecializationConstants {
        SpecializationConstants {}
    }
}
unsafe impl SpecConstsTrait for SpecializationConstants {
    fn descriptors() -> &'static [SpecializationMapEntry] {
        static DESCRIPTORS: [SpecializationMapEntry; 0usize] = [];
        &DESCRIPTORS
    }
}
