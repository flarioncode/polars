use std::any::Any;
use std::sync::mpsc;

use arrow::legacy::error::PolarsResult;
use polars_plan::prelude::FlarionChannelMessage;

use crate::operators::{DataChunk, FinalizedSink, PExecutionContext, Sink, SinkResult};

// Currently ignore the index in the DataChunk
// When we add sort support we'll need to ensure we get them in the right order
#[derive(Clone)]
pub struct ChannelSink {
    _partition_id: i32,
    flarion_channel_tx: mpsc::Sender<FlarionChannelMessage>,
}

impl ChannelSink {
    pub fn new(partition_id: i32, flarion_channel_tx: mpsc::Sender<FlarionChannelMessage>) -> Self {
        Self {
            _partition_id: partition_id,
            flarion_channel_tx,
        }
    }
}

impl Sink for ChannelSink {
    fn sink(&mut self, _context: &PExecutionContext, chunk: DataChunk) -> PolarsResult<SinkResult> {
        // don't add empty dataframes
        if chunk.data.height() > 0 {
            let stored_msg = FlarionChannelMessage::DataReady(chunk.data);
            self.flarion_channel_tx
                .send(stored_msg)
                .expect("Could not send message");
            // loop {
            //     match POOL.install(|| self.flarion_channel_tx.send(stored_msg)) {
            //         Ok(_) => break,
            //         Err(mpsc::TrySendError::Full(returned_msg)) => {
            //             stored_msg = returned_msg;
            //         },
            //         Err(mpsc::TrySendError::Disconnected(_)) => {
            //             panic!("Channel disconnected");
            //         }
            //     }
            // }
        };
        Ok(SinkResult::CanHaveMoreInput)
    }

    fn combine(&mut self, _other: &mut dyn Sink) {
        // ignore this for now
    }

    fn split(&self, _thread_no: usize) -> Box<dyn Sink> {
        Box::new(self.clone())
    }

    fn finalize(&mut self, _context: &PExecutionContext) -> PolarsResult<FinalizedSink> {
        // `Depleted` indicates that we can flush all remaining chunks.
        self.flarion_channel_tx
            .send(FlarionChannelMessage::Depleted)
            .unwrap();

        // return a dummy dataframe;
        Ok(FinalizedSink::Finished(Default::default()))
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }

    fn fmt(&self) -> &str {
        "channel_sink"
    }
}
