using FastBertTokenizer;

namespace SharpEmbed;

public interface IEmbeddings
{
    public float[] ComputeForSentence(string sentence);
}

public class Embeddings : IEmbeddings
{
    // TODO abstract away the tokenizer and demand an interface
    private readonly BertTokenizer _tokenizer;

    public Embeddings(BertTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    public float[] ComputeForSentence(string sentence)
    {
        var (inputIds, attentionMask, tokenTypeIds) = _tokenizer.Encode(sentence);
        // TODO implement me :-)
        return Array.Empty<float>();
    }
}