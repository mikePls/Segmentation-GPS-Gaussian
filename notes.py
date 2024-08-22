

# Save raw mask
tmp_mask = logits_left[0].detach().sigmoid()
tmp_mask_raw = tmp_mask.permute(1, 2, 0).cpu().numpy() * 255
tmp_mask_raw_name = '%s/%s_mask_raw.jpg' % (cfg.record.show_path, self.total_steps)
cv2.imwrite(tmp_mask_raw_name, tmp_mask_raw.astype(np.uint8))

# Apply threshold to the mask and save it
threshold = 0.5
tmp_mask_thresholded = (tmp_mask > threshold).float() * 255
tmp_mask_thresholded = tmp_mask_thresholded.permute(1, 2, 0).cpu().numpy()
tmp_mask_thresholded_name = '%s/%s_mask_thresholded.jpg' % (cfg.record.show_path, self.total_steps)
cv2.imwrite(tmp_mask_thresholded_name, tmp_mask_thresholded.astype(np.uint8))

# Save the corresponding image
tmp_image = data['lmain']['img'][0].detach().permute(1, 2, 0).cpu().numpy()
# De-normalize the image if it was normalized to [-1, 1]
tmp_image = ((tmp_image + 1) * 127.5).astype(np.uint8)
tmp_image_name = '%s/%s_image.jpg' % (cfg.record.show_path, self.total_steps)
cv2.imwrite(tmp_image_name, tmp_image[:, :, ::-1])

for view in ['lmain', 'rmain']:
    valid = (data[view]['valid'] >= 0.5)
    epe = torch.sum((data[view]['flow'] - data[view]['flow_pred']) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    one_pix = (epe < 1)
    epe_list.append(epe.mean().item())
    one_pix_list.append(one_pix.float().mean().item())