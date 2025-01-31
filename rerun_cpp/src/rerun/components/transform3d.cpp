// NOTE: This file was autogenerated by re_types_builder; DO NOT EDIT.
// Based on "crates/re_types/definitions/rerun/components/transform3d.fbs"

#include "transform3d.hpp"

#include "../arrow.hpp"
#include "../datatypes/transform3d.hpp"

#include <arrow/api.h>

namespace rerun {
    namespace components {
        const char *Transform3D::NAME = "rerun.transform3d";

        const std::shared_ptr<arrow::DataType> &Transform3D::to_arrow_datatype() {
            static const auto datatype = rerun::datatypes::Transform3D::to_arrow_datatype();
            return datatype;
        }

        arrow::Result<std::shared_ptr<arrow::DenseUnionBuilder>>
            Transform3D::new_arrow_array_builder(arrow::MemoryPool *memory_pool) {
            if (!memory_pool) {
                return arrow::Status::Invalid("Memory pool is null.");
            }

            return arrow::Result(
                rerun::datatypes::Transform3D::new_arrow_array_builder(memory_pool).ValueOrDie()
            );
        }

        arrow::Status Transform3D::fill_arrow_array_builder(
            arrow::DenseUnionBuilder *builder, const Transform3D *elements, size_t num_elements
        ) {
            if (!builder) {
                return arrow::Status::Invalid("Passed array builder is null.");
            }
            if (!elements) {
                return arrow::Status::Invalid("Cannot serialize null pointer to arrow array.");
            }

            static_assert(sizeof(rerun::datatypes::Transform3D) == sizeof(Transform3D));
            ARROW_RETURN_NOT_OK(rerun::datatypes::Transform3D::fill_arrow_array_builder(
                builder,
                reinterpret_cast<const rerun::datatypes::Transform3D *>(elements),
                num_elements
            ));

            return arrow::Status::OK();
        }

        arrow::Result<rerun::DataCell> Transform3D::to_data_cell(
            const Transform3D *instances, size_t num_instances
        ) {
            // TODO(andreas): Allow configuring the memory pool.
            arrow::MemoryPool *pool = arrow::default_memory_pool();

            ARROW_ASSIGN_OR_RAISE(auto builder, Transform3D::new_arrow_array_builder(pool));
            if (instances && num_instances > 0) {
                ARROW_RETURN_NOT_OK(
                    Transform3D::fill_arrow_array_builder(builder.get(), instances, num_instances)
                );
            }
            std::shared_ptr<arrow::Array> array;
            ARROW_RETURN_NOT_OK(builder->Finish(&array));

            auto schema = arrow::schema(
                {arrow::field(Transform3D::NAME, Transform3D::to_arrow_datatype(), false)}
            );

            rerun::DataCell cell;
            cell.component_name = Transform3D::NAME;
            ARROW_ASSIGN_OR_RAISE(
                cell.buffer,
                rerun::ipc_from_table(*arrow::Table::Make(schema, {array}))
            );

            return cell;
        }
    } // namespace components
} // namespace rerun
