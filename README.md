# Tera
Tera is an AI assistant which is tailored just for you and runs fully locally.

> I just published a [Medium Article](https://medium.com/@isalapiyarisi/lets-build-a-standalone-chatbot-with-phi-2-and-rust-48c0f714f915) about how I built Tera. Check it out if you're interested.

## Build

1. Make sure you have all the dependencies installed.
    - [Rust](https://www.rust-lang.org/tools/install)
    - [ffmpeg](https://ffmpeg.org/)
    - [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html)
2. Clone the repository
    ```bash
    git clone https://github.com/MrSupiri/Tera
    ```
3. Build the project
    ```bash
    cargo build --release
    ```

4. Use Tera
    ```bash
    cargo run --release -- remember "Naruto's favorite ramen is miso-flavored."
    ```


## Usage

> As of now only external dependency that is needed by Tera is [ffmpeg](https://ffmpeg.org/). It is used to convert audio files to wav format. So if you want to upload audio files to Tera you need to have ffmpeg installed.


```bash
Usage: tera <COMMAND>

Commands:
  ask       Ask a question
  upload    Let Tera learn from your content
  remember  Tell Tera something to remember
  forget    Forget something Tera remembers
  list      List all content Tera remembers sorted by added date
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

## Use Cases

1. **Personalized Learning**: Tera can help you learn new topics by asking it to remember key facts, then quizzing you later.
2. **Memory Aid**: Tell Tera important things you need to remember, like birthdays, anniversaries, or grocery lists. It can remind you when needed.
3. **Mental Health Companion**: Tell Tera about your feelings or thoughts. It can provide comforting words, motivational quotes, or mindfulness exercises.
4. **Motivational Coach**: Feed Tera with motivational quotes and ask it to inspire you when you need a boost.
4. **Home Inventory Manager**: Tell Tera about your home inventory and ask it to remind you when you're running low on supplies.

### TODOs
- [ ] Make use of SurrealDB's [vector indexing](https://www.youtube.com/watch?v=2MmyE_iohEs) to improve performance once it's available.
- [ ] Publish prebuilt binaries.
- [ ] Add CUDA and Metal support for faster inference.
- [ ] Remove ffmpeg dependency.

## Licence
[AGPL-3.0-or-later](LICENSE)
