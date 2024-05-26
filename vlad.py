import torch
import utils
import copy

def vlad_score(video, ar_model, clip_model, clip_processor, label_names):
	print('***************************')
	print('VLAD Score calculation')

	# ar_model score
	out = ar_model(video)
	post_act = torch.nn.Softmax(dim=1)
	out = post_act(out)
	top5_scores, top5_labels = torch.topk(out, 5)

	predicted_label = top5_labels[0][0]
	print('ar_model predicted label: ' + str(predicted_label))

	model_scores = out[0]

	video_to_clip = copy.deepcopy(video)
	video_to_clip = video_to_clip * 255
	video_to_clip = video_to_clip.to(torch.uint8)
	video_to_clip = torch.squeeze(video_to_clip).permute(1, 2, 3, 0).to('cpu').detach().numpy()

	# clip score
	with torch.no_grad():
		inputs = clip_processor(text=label_names, images=video_to_clip, return_tensors="pt", padding=True)

		inputs['pixel_values'] = inputs['pixel_values'].to('cuda')
		inputs['input_ids'] = inputs['input_ids'].to('cuda')
		inputs['attention_mask'] = inputs['attention_mask'].to('cuda')

		outputs = clip_model(**inputs)
		logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
		probs = logits_per_image.softmax(dim=1)

	mean_probs = torch.mean(probs, dim=0)
	utils.summary(torch.unsqueeze(mean_probs, dim=0), label_names)
	clip_true_label_score = mean_probs[predicted_label]
	print('clip probability for ar_model predicted label: ' + str(clip_true_label_score))

	kl1 = utils.kl_divergence(mean_probs, model_scores)
	kl2 = utils.kl_divergence(model_scores, mean_probs)
	kl = (kl1 + kl2) / 2

	print('KL divergence: ' + str(kl))
	print('***************************')
	return kl.item()
