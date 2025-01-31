// NOTE: This file was autogenerated by re_types_builder; DO NOT EDIT.
// Based on "crates/re_types/definitions/rerun/datatypes/angle.fbs"

#include "angle.hpp"

#include <arrow/api.h>

namespace rerun {
    namespace datatypes {
        const std::shared_ptr<arrow::DataType> &Angle::to_arrow_datatype() {
            static const auto datatype = arrow::dense_union({
                arrow::field("_null_markers", arrow::null(), true, nullptr),
                arrow::field("Radians", arrow::float32(), false),
                arrow::field("Degrees", arrow::float32(), false),
            });
            return datatype;
        }

        arrow::Result<std::shared_ptr<arrow::DenseUnionBuilder>> Angle::new_arrow_array_builder(
            arrow::MemoryPool *memory_pool
        ) {
            if (!memory_pool) {
                return arrow::Status::Invalid("Memory pool is null.");
            }

            return arrow::Result(std::make_shared<arrow::DenseUnionBuilder>(
                memory_pool,
                std::vector<std::shared_ptr<arrow::ArrayBuilder>>({
                    std::make_shared<arrow::NullBuilder>(memory_pool),
                    std::make_shared<arrow::FloatBuilder>(memory_pool),
                    std::make_shared<arrow::FloatBuilder>(memory_pool),
                }),
                to_arrow_datatype()
            ));
        }

        arrow::Status Angle::fill_arrow_array_builder(
            arrow::DenseUnionBuilder *builder, const Angle *elements, size_t num_elements
        ) {
            if (!builder) {
                return arrow::Status::Invalid("Passed array builder is null.");
            }
            if (!elements) {
                return arrow::Status::Invalid("Cannot serialize null pointer to arrow array.");
            }

            ARROW_RETURN_NOT_OK(builder->Reserve(static_cast<int64_t>(num_elements)));
            for (size_t elem_idx = 0; elem_idx < num_elements; elem_idx += 1) {
                const auto &union_instance = elements[elem_idx];
                ARROW_RETURN_NOT_OK(builder->Append(static_cast<int8_t>(union_instance._tag)));

                auto variant_index = static_cast<int>(union_instance._tag);
                auto variant_builder_untyped = builder->child_builder(variant_index).get();

                switch (union_instance._tag) {
                    case detail::AngleTag::NONE: {
                        ARROW_RETURN_NOT_OK(variant_builder_untyped->AppendNull());
                        break;
                    }
                    case detail::AngleTag::Radians: {
                        auto variant_builder =
                            static_cast<arrow::FloatBuilder *>(variant_builder_untyped);
                        ARROW_RETURN_NOT_OK(variant_builder->Append(union_instance._data.radians));
                        break;
                    }
                    case detail::AngleTag::Degrees: {
                        auto variant_builder =
                            static_cast<arrow::FloatBuilder *>(variant_builder_untyped);
                        ARROW_RETURN_NOT_OK(variant_builder->Append(union_instance._data.degrees));
                        break;
                    }
                }
            }

            return arrow::Status::OK();
        }
    } // namespace datatypes
} // namespace rerun
