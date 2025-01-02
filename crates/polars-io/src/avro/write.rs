use std::io::Write;

pub use arrow::io::avro::avro_schema::file::Compression;
use arrow::io::avro::avro_schema::{self};
use arrow::io::avro::write;
use polars_core::error::to_compute_err;
use polars_core::prelude::*;
pub use Compression as AvroCompression;
use arrow::io::avro::avro_schema::file::CompressedBlock;
use crate::shared::{schema_to_arrow_checked, SerWriter};

/// Write a [`DataFrame`] to [Apache Avro] format
///
/// [Apache Avro]: https://avro.apache.org
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::avro::AvroWriter;
/// use std::fs::File;
/// use polars_io::SerWriter;
///
/// fn example(df: &mut DataFrame) -> PolarsResult<()> {
///     let mut file = File::create("file.avro").expect("could not create file");
///
///     AvroWriter::new(&mut file)
///         .finish(df)
/// }
/// ```
#[must_use]
pub struct AvroWriter<W> {
    writer: W,
    compression: Option<AvroCompression>,
    name: String,
}

impl<W> AvroWriter<W>
where
    W: Write,
{
    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<AvroCompression>) -> Self {
        self.compression = compression;
        self
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
}

#[inline]
fn write_blocks(df: &DataFrame, record: &avro_schema::schema::Record, writer: &mut impl Write, data: &mut Vec<u8>, compressed_block: &mut CompressedBlock, compression: Option<AvroCompression>) -> PolarsResult<()> {
    for chunk in df.iter_chunks(CompatLevel::oldest(), true) {
        let mut serializers = chunk
            .iter()
            .zip(record.fields.iter())
            .map(|(array, field)| write::new_serializer(array.as_ref(), &field.schema))
            .collect::<Vec<_>>();

        let mut block =
            avro_schema::file::Block::new(chunk.arrays()[0].len(), std::mem::take(data));
        write::serialize(&mut serializers, &mut block);
        let _was_compressed =
            avro_schema::write::compress(&mut block, compressed_block, compression)
                .map_err(to_compute_err)?;

        avro_schema::write::write_metadata(writer, record.clone(), compression)
            .map_err(to_compute_err)?;

        avro_schema::write::write_block(writer, compressed_block)
            .map_err(to_compute_err)?;
        // reuse block for next iteration.
        *data = block.data;
        data.clear();

        // reuse block for next iteration
        compressed_block.data.clear();
        compressed_block.number_of_rows = 0
    }

    Ok(())
}

impl<W> SerWriter<W> for AvroWriter<W>
where
    W: Write,
{
    fn new(writer: W) -> Self {
        Self {
            writer,
            compression: None,
            name: "".to_string(),
        }
    }

    fn finish(&mut self, df: &mut DataFrame) -> PolarsResult<()> {
        let schema = schema_to_arrow_checked(&df.schema(), CompatLevel::oldest(), "avro")?;
        let record = write::to_record(&schema, self.name.clone())?;

        let mut data = vec![];
        let mut compressed_block = CompressedBlock::default();
        write_blocks(df, &record, &mut self.writer, &mut data, &mut compressed_block, self.compression)
    }
}


pub struct BatchedAvroWriter<W> {
    schema: Option<(ArrowSchema, avro_schema::schema::Record)>,
    data: Vec<u8>,
    compressed_block: CompressedBlock,
    writer: W,
    compression: Option<AvroCompression>,
    name: String,
}

impl<W> BatchedAvroWriter<W>
where
    W: Write,
{
    pub fn new(writer: W) -> Self {
        Self {
            schema: None,
            data: vec![],
            compressed_block: CompressedBlock::default(),
            writer,
            compression: None,
            name: "".to_string(),
        }
    }

    /// Set the compression used. Defaults to None.
    pub fn with_compression(mut self, compression: Option<AvroCompression>) -> Self {
        self.compression = compression;
        self
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn write_batch(&mut self, df: &DataFrame) -> PolarsResult<()> {
        let new_schema = schema_to_arrow_checked(&df.schema(), CompatLevel::oldest(), "avro")?;
        let record = match self.schema.as_ref() {
            Some(schema) => {
                assert_eq!(new_schema, schema.0);
                &schema.1
            }
            None => {
                let avro_schema = write::to_record(&new_schema, self.name.clone())?;
                self.schema = Some((new_schema, avro_schema));
                &self.schema.as_ref().unwrap().1
            }
        };


        write_blocks(df, record, &mut self.writer, &mut self.data, &mut self.compressed_block, self.compression)
    }
}
