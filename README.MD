# StreamSpeech - Real-time Speech-to-Speech Translation

## Description

StreamSpeech is a real-time speech-to-speech translation system designed to perform simultaneous speech recognition, translation, and synthesis. Leveraging state-of-the-art Conformer models for encoding, LSTM-based decoders for text generation, and HiFi-GAN vocoders for speech synthesis, StreamSpeech provides an efficient and integrated solution for real-time communication needs.

This project includes data preparation, model training, and inference scripts, enabling seamless conversion from input speech in one language to output speech in another language with minimal latency.

## Features

- **Simultaneous Speech Recognition and Translation**: Uses advanced Conformer encoders and LSTM decoders to handle streaming inputs and generate accurate translations in real-time.
- **High-Quality Speech Synthesis**: Utilizes HiFi-GAN vocoders to produce natural and high-fidelity speech outputs.
- **Unified Multi-task Learning**: Integrates multiple tasks into a single model, optimizing performance and reducing the complexity of separate modules.
- **Flexible and Scalable**: Easily adapt the model to different languages and applications by training on relevant datasets.

## Use Cases

- Real-time translation for international conferences and meetings.
- Live subtitle generation for broadcasts and online streams.
- Communication aids for multilingual conversations in various settings.
