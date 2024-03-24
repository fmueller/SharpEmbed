using FastBertTokenizer;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SharpEmbed;

public interface IEmbeddings
{
    public float[] ComputeForSentence(string sentence);
}

public class Embeddings : IEmbeddings
{
    private readonly string _pathToModel;
    // TODO abstract away the tokenizer and demand an interface
    private readonly BertTokenizer _tokenizer;

    public Embeddings(string pathToModel, BertTokenizer tokenizer)
    {
        _pathToModel = pathToModel;
        _tokenizer = tokenizer;
    }

    public float[] ComputeForSentence(string sentence)
    {
        var (inputIds, attentionMask, tokenTypeIds) = _tokenizer.Encode(sentence);
        var inputs = new List<NamedOnnxValue>
        {
            CreateTensor("input_ids", inputIds.ToArray()),
            CreateTensor("attention_mask", attentionMask.ToArray()),
            CreateTensor("token_type_ids", tokenTypeIds.ToArray())
        };

        using var session = new InferenceSession(_pathToModel);
        var lastHiddenState = session.Run(inputs).First(item => item.Name == "last_hidden_state").AsTensor<float>();
        return MeanPooling(lastHiddenState, attentionMask);
    }

    private static NamedOnnxValue CreateTensor(string name, long[] data)
    {
        return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, new[] { 1, data.Length }));
    }

    private static float[] MeanPooling(Tensor<float> lastHiddenState, ReadOnlyMemory<long> attentionMask)
    {
        int batchSize = lastHiddenState.Dimensions[0];
        int seqLength = lastHiddenState.Dimensions[1];
        int embedDim = lastHiddenState.Dimensions[2];
        float[] returnedData = new float[batchSize * embedDim];

        int outIndex = 0;
        for (int i = 0; i < batchSize; i++)
        {
            int offset = i * embedDim * seqLength;

            for (int k = 0; k < embedDim; k++)
            {
                float sum = 0;
                int count = 0;

                int attnMaskOffset = i * seqLength;
                int offset2 = offset + k;
                // Pool over all words in sequence
                for (int j = 0; j < seqLength; j++)
                {
                    // index into attention mask
                    long attn = attentionMask.Span[attnMaskOffset + j];

                    count += (int)attn;
                    float kValue = lastHiddenState.GetValue(offset2 + j * embedDim);
                    sum += kValue * attn;
                }

                float avg = sum / count;
                returnedData[outIndex++] = avg;
            }
        }

        return returnedData;
    }
}