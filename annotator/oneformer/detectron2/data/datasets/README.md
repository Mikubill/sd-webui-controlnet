

### Common Datasets

The datasets implemented here do not need to load the data into the final format.
Each dataset should provide the minimal data structure needed to use the dataset, so it can be very efficient.

For example, for an image dataset, just provide the file names and labels, but don't read the images.
Let downstream decide how to read.
