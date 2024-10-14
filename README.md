# timbre

Timbre remains an “unsolved problem” in that there’s practically no largely agreeable standard for how to describe or represent it. We tend to either use spectrograms or natural language, both which suck. Historically, we have subjectively organized timbres using perceptual surveys. These are at-best intuitive, but are discrete and limited to the subset of sounds they include. Now, we bias ourselves in wanting to “solve this problem” using machine learning and enough data.

## The Ideal Model
> A timbre space organized by the most salient perceptual properties (”principal components”) as a bi-directional, joint text-audio*, generative model

This space enables us to:

- **Analyze**: observe how timbres perceptually relate to each other via spatial relationships.
- **Generate**: hear what a timbre sounds like at any point in this space (similar to timbre interpolation).
- **Interpret**: describe a timbre in words and what a timbre would sound like given words.
    
> *We will not dare try incorporating text into our model yet. Unless I write otherwise, assume all further details are for an audio-only model.

## Development Setup

> [TODO] Containerize and standardize setup

We do not know the minimally support Python version, but we are mostly using `>=3.11`. 

Download requirements via `pip install -r malleus_requirements.txt`, although be aware we manually maintain this frozen list.
