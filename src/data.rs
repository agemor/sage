use std::fs;
use std::io;
use std::io::{Read, SeekFrom, Seek, Error};

pub trait Dataset: IntoIterator {}

pub struct DatasetIterator {}

impl Iterator for DatasetIterator {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

pub struct Loader {}

impl Loader {
    pub fn new(dataset: Box<dyn Dataset>, batch_size: usize) -> Loader {
        Loader {}
    }
}



pub struct Mnist {
    images: Vec<[u8; Mnist::IMAGE_SIZE]>,
    labels: Vec<u8>,
}

impl Mnist {
    const IMAGE_SIZE: usize = 28 * 28;

    pub fn from_source(image_path: &str, label_path: &str) -> io::Result<Mnist> {
        let images = Self::read_images(image_path)?;
        let labels = Self::read_labels(label_path)?;

        Ok(Mnist { images, labels })
    }

    fn read_images(path: &str) -> io::Result<Vec<[u8; Self::IMAGE_SIZE]>> {
        let mut f = fs::File::open(path)?;

        let mut buf_32: [u8; 4] = [0; 4];

        f.seek(SeekFrom::Start(4))?;
        f.read_exact(&mut buf_32)?;
        f.seek(SeekFrom::Current(8))?;

        let num_images = u32::from_be_bytes(buf_32);

        let mut images: Vec<[u8; Self::IMAGE_SIZE]> = Vec::with_capacity(num_images as usize);
        let mut buffer_image: [u8; Self::IMAGE_SIZE] = [0; Self::IMAGE_SIZE];

        for _ in 0..num_images {
            f.read_exact(&mut buffer_image)?;
            images.push(buffer_image.clone());
        }
        Ok(images)
    }

    fn read_labels(path: &str) -> io::Result<Vec<u8>> {
        let mut f = fs::File::open(path)?;

        let mut buf_8: [u8; 1] = [0; 1];
        let mut buf_32: [u8; 4] = [0; 4];

        f.seek(SeekFrom::Start(4))?;
        f.read_exact(&mut buf_32)?;

        let num_labels = u32::from_be_bytes(buf_32);

        let mut labels: Vec<u8> = Vec::with_capacity(num_labels as usize);

        for _ in 0..num_labels {
            f.read_exact(&mut buf_8)?;
            labels.push(buf_8[0]);
        }
        Ok(labels)
    }
}
