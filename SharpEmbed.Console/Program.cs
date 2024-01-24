using System.CommandLine;
using FastBertTokenizer;

namespace SharpEmbed.Console;

using Console = System.Console;

class Program
{
    static async Task<int> Main(string[] args)
    {
        var inputOption = new Option<string>(
            name: "--input",
            description: "The text to compute embeddings for.");

        var modelOption = new Option<string>(
            name: "--model",
            description: "HuggingFace model to use.",
            getDefaultValue: () => "bert-base-uncased");

        var rootCommand = new RootCommand("SharpEmbed computes embeddings for your text.");
        rootCommand.AddOption(inputOption);
        rootCommand.AddOption(modelOption);
        rootCommand.SetHandler((text, model) => { ComputeEmbeddings(text, model).Wait(); }, inputOption, modelOption);

        return await rootCommand.InvokeAsync(args);
    }

    private static async Task ComputeEmbeddings(string text, string model)
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

        var sentenceEmbeddings = new Embeddings(tokenizer).ComputeForSentence(text);
        Console.WriteLine($"sentenceEmbeddings: {string.Join(", ", sentenceEmbeddings)}");
    }
}