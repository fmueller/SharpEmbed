using System.CommandLine;
using FastBertTokenizer;

namespace SharpEmbed.Cli;

class Program
{
    static async Task<int> Main(string[] args)
    {
        var inputOption = new Option<string>(
            name: "--input",
            description: "The text to compute embeddings for.");

        var modelOption = new Option<string>(
            name: "--model",
            description: "HuggingFace model to use for tokenizer.",
            getDefaultValue: () => "bert-base-uncased");

        var onnxModelOption = new Option<string>(
            name: "--onnx-model",
            description: "HuggingFace model to use to compute embeddings.",
            getDefaultValue: () => "bert-base-uncased");

        var rootCommand = new RootCommand("SharpEmbed computes embeddings for your text.");
        rootCommand.AddOption(inputOption);
        rootCommand.AddOption(modelOption);
        rootCommand.AddOption(onnxModelOption);
        rootCommand.SetHandler((text, model, onnxModel) => { ComputeEmbeddings(text, model, onnxModel).Wait(); },
            inputOption, modelOption, onnxModelOption);

        return await rootCommand.InvokeAsync(args);
    }

    private static async Task ComputeEmbeddings(string text, string model, string onnxModelPath)
    {
        Console.WriteLine($"Computing embeddings for '{text}' using model '{model}'...");
        var tokenizer = new BertTokenizer();
        await tokenizer.LoadFromHuggingFaceAsync(model);

        var (inputIds, attentionMask, tokenTypeIds) = tokenizer.Encode(text);
        Console.WriteLine($"Input IDs: {string.Join(", ", inputIds.ToArray())}");
        Console.WriteLine($"Attention Mask: {string.Join(", ", attentionMask.ToArray())}");
        Console.WriteLine($"Token Type IDs: {string.Join(", ", tokenTypeIds.ToArray())}");

        var decoded = tokenizer.Decode(inputIds.Span);
        Console.WriteLine($"Decoded: {decoded}");

        var embeddings = new Embeddings(onnxModelPath, tokenizer);

        var sentenceEmbeddings = embeddings.ComputeForSentence(text);
        Console.WriteLine($"Embeddings: {string.Join(", ", sentenceEmbeddings)}");
        Console.WriteLine($"Embeddings Size: {sentenceEmbeddings.Length}");

        var otherSentenceEmbeddings = embeddings.ComputeForSentence(text);
        var similarity = embeddings.CosineSimilarity(sentenceEmbeddings, otherSentenceEmbeddings);
        Console.WriteLine($"Similarity (should be 1): {similarity}");
    }
}