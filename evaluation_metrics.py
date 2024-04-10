import torch
from tqdm import tqdm


def evaluate_next_token_accuracy(model, dataloader, device):
    # Encode the input text
    corrects = 0
    total_masked = 0

    for i, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(device)
        masked_input = batch.clone()
        with torch.no_grad():
            logits = model(masked_input, labels=batch).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        predicted_token_id = shift_logits.argmax(axis=-1)
        corrects += (shift_labels == predicted_token_id).sum().item()
        total_masked += predicted_token_id.shape[0] * predicted_token_id.shape[1]

    return corrects / total_masked


def evaluate_mask_prediction_accuracy(model, dataloader, device, mask_percentage=0.3):
    corrects = 0
    total_masked = 0
    for i, batch in enumerate(tqdm(dataloader)):
        batch = batch.to(device)
        masked_input = batch.clone()
        random_mask = torch.rand(masked_input.shape).to(device)  # I can only do this for non-overlapping
        random_mask = (random_mask < mask_percentage) * (masked_input != 0)  # Cannot mask the [<UNK>] token
        mask_idx = (random_mask.flatten() == True).nonzero().view(-1)
        masked_input = masked_input.flatten()
        masked_input[mask_idx] = 1
        masked_input = masked_input.view(batch.size())
        with torch.no_grad():
            predictions = model(masked_input, labels=batch).logits
        # Get the predicted token id
        predicted_token_id = predictions.argmax(axis=-1)

        grand_truth = batch.flatten()[mask_idx]
        prediction = predicted_token_id.flatten()[mask_idx]

        corrects += (grand_truth == prediction).sum().item()
        total_masked += len(mask_idx)
    return corrects / total_masked


if __name__ == '__main__':
    # text = "The quick brown fox jumps over the lazy"
    # next_word = "dog"
    # accuracy = evaluate_accuracy(model, tokenizer, text, next_word)
    # print("Correct Prediction:" if accuracy else "Wrong Prediction:", accuracy)
    pass
