#include "util.hpp"

void util::print_tensor_shape(const TfLiteTensor *tensor)
{
    printf("[");
    for (int i = 0; i < tensor->dims->size; ++i)
    {
        printf("%d", tensor->dims->data[i]);
        if (i < tensor->dims->size - 1)
            printf(", ");
    }
    printf("]");
}

void util::print_model_summary(tflite::Interpreter *interpreter, bool delegate_applied)
{
    printf("\n[INFO] Model Summary \n");
    printf("📥 Input tensor count  : %zu\n", interpreter->inputs().size());
    printf("📤 Output tensor count : %zu\n", interpreter->outputs().size());
    printf("📦 Total tensor count  : %ld\n", interpreter->tensors_size());
    printf("🔧 Node (op) count     : %zu\n", interpreter->nodes_size());
    printf("🧩 Delegate applied    : %s\n", delegate_applied ? "Yes ✅" : "No ❌");
}

// Get indices of top-k highest values
std::vector<int> util::get_topK_indices(const std::vector<float> &data, int k)
{
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&data](int a, int b)
        { return data[a] > data[b]; });
    indices.resize(k);
    return indices;
}

// Load label file from JSON and return index → label map
std::unordered_map<int, std::string> util::load_class_labels(const std::string &json_path)
{
    std::ifstream ifs(json_path, std::ifstream::binary);
    if (!ifs.is_open())
        throw std::runtime_error("Failed to open label file: " + json_path);

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    if (!Json::parseFromStream(builder, ifs, &root, &errs))
        throw std::runtime_error("Failed to parse JSON: " + errs);

    std::unordered_map<int, std::string> label_map;

    for (const auto &key : root.getMemberNames())
    {
        int idx = std::stoi(key);
        if (root[key].isArray() && root[key].size() >= 2)
        {
            label_map[idx] = root[key][1].asString(); // label = second element
        }
    }

    return label_map;
}

//**** For Section  2.4 ****/

void util::timer_start(const std::string &label)
{
    util::timers[label] = Clock::now();
}

void util::timer_stop(const std::string &label)
{
    auto end = Clock::now();
    if (util::timers.find(label) != util::timers.end())
    {
        auto start = timers[label];
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[TIMER] " << label << " took " << duration << " ms" << std::endl;
        timers.erase(label); // cleanup
    }
    else
    {
        std::cerr << "[TIMER] No start time recorded for label: " << label << std::endl;
    }
}
//*==========================================*/
