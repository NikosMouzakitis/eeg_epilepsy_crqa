for i in {1..24}; do
  rclone mkdir drive:EEG_Processed/pat$i/normal
  rclone mkdir drive:EEG_Processed/pat$i/epileptic
done


