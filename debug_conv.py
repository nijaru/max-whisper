#!/usr/bin/env python3
import whisper

model = whisper.load_model("tiny")
conv1 = model.encoder.conv1
conv2 = model.encoder.conv2

print("Conv1 weight shape:", conv1.weight.shape)
print("Conv1 bias shape:", conv1.bias.shape) 
print("Conv2 weight shape:", conv2.weight.shape)
print("Conv2 bias shape:", conv2.bias.shape)

print("\nConv1 details:")
print("- in_channels:", conv1.in_channels)
print("- out_channels:", conv1.out_channels)
print("- kernel_size:", conv1.kernel_size)
print("- stride:", conv1.stride)
print("- padding:", conv1.padding)

print("\nConv2 details:")
print("- in_channels:", conv2.in_channels)
print("- out_channels:", conv2.out_channels)
print("- kernel_size:", conv2.kernel_size)
print("- stride:", conv2.stride)
print("- padding:", conv2.padding)