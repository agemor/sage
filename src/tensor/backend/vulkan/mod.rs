use crate::tensor::backend;
use crate::tensor::backend::vulkan::processor::Processor;
use std::rc::Rc;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::device::{Device, DeviceExtensions, Queue, QueuesIter};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};

#[macro_use]
mod processor;
mod shader;

#[derive(Clone)]
pub struct Backend {
    device: Arc<Device>,
    processor: Rc<Processor>,
}

impl Backend {
    pub fn new() -> Self {
        // create instance
        let app_infos = app_info_from_cargo_toml!();
        let instance = Instance::new(Some(&app_infos), &InstanceExtensions::none(), None)
            .expect("No Vulkan implementations available.");

        // create device
        let (device, mut queues) = Self::get_device(&instance);
        let queue = queues.next().unwrap();

        let processor = Rc::new(Processor::new(device.clone(), queue));

        let backend = Backend { device, processor };

        backend
    }

    fn get_device(instance: &Arc<Instance>) -> (Arc<Device>, QueuesIter) {
        // select device at index 0
        let physical_device = PhysicalDevice::from_index(&instance, 0)
            .expect("Failed to get the specified physical device.");

        let queue_family = physical_device
            .queue_families()
            .find(|&q| q.supports_compute())
            .expect("Couldn't find a compute queue family.");

        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        };

        Device::new(
            physical_device,
            physical_device.supported_features(),
            &device_extensions,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    }
}

impl backend::Backend for Backend {
    fn alloc_mem(&self, size: usize) -> Rc<dyn backend::Buffer<Backend = Backend>> {
        Rc::new(Buffer::new(size, self.clone()))
    }

    fn alloc_mem_from_iter<I>(&self, data: I) -> Rc<dyn backend::Buffer<Backend = Backend>>
    where
        I: ExactSizeIterator<Item = f32>,
    {
        Rc::new(Buffer::from_iter(data, self.clone()))
    }

    fn processor(&self) -> &dyn backend::Processor {
        self.processor.as_ref()
    }
}

pub struct Buffer {
    backend: Backend,
    data: Arc<CpuAccessibleBuffer<[f32]>>,
}

impl Buffer {
    pub fn new(size: usize, backend: Backend) -> Self {
        unsafe {
            Buffer {
                backend,
                data: CpuAccessibleBuffer::uninitialized_array(
                    backend.device.clone(),
                    size,
                    BufferUsage::all(),
                    false,
                )
                .unwrap(),
            }
        }
    }

    pub fn from_iter<I>(data: I, backend: Backend) -> Self
    where
        I: ExactSizeIterator<Item = f32>,
    {
        Buffer {
            backend,
            data: CpuAccessibleBuffer::from_iter(
                backend.device.clone(),
                BufferUsage::all(),
                false,
                data,
            )
            .unwrap(),
        }
    }

    pub fn downcast(buffer: &dyn backend::Buffer<Backend = Backend>) -> &Self {
        buffer.as_any().downcast_ref::<Self>().unwrap()
    }
}
