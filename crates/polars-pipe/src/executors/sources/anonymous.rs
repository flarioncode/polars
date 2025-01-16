use std::sync::Arc;

use arrow::legacy::error::PolarsResult;
use polars_plan::prelude::{AnonymousScan, AnonymousScanArgs, AnonymousScanOptions};
use polars_utils::IdxSize;

use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};

pub struct AnonymousSource {
    function: Arc<dyn AnonymousScan>,
    args: AnonymousScanArgs,
    current_idx: IdxSize,
}

impl AnonymousSource {
    pub fn new(_options: Arc<AnonymousScanOptions>, function: Arc<dyn AnonymousScan>) -> Self {
        Self {
            args: AnonymousScanArgs {
                n_rows: None,
                with_columns: None,
                schema: function.schema(None).unwrap(),
                output_schema: None,
                predicate: None,
            },
            function,
            current_idx: 0,
        }
    }
}

impl Source for AnonymousSource {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        if let Some(batch) = self.function.next_batch(&self.args)? {
            let res = SourceResult::GotMoreData(vec![DataChunk::new(self.current_idx, batch)]);
            self.current_idx += 1;
            Ok(res)
        } else {
            Ok(SourceResult::Finished)
        }
    }

    fn fmt(&self) -> &str {
        "anonymous"
    }
}
