from trainer.unlearn.base import UnlearnTrainer


class GradAscent(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)
        forget_loss_original = outputs.loss.item()
        loss = -outputs.loss
        
        # Log metrics for monitoring
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "forget_loss_original": forget_loss_original,
                "forget_loss_negated": loss.item(),
            })
        
        return (loss, outputs) if return_outputs else loss
