// Log a simple line strip.

#include <rerun.hpp>

namespace rr = rerun;

int main() {
    auto rr_stream = rr::RecordingStream("line_strip3d");
    rr_stream.connect("127.0.0.1:9876");

    std::vector<rr::datatypes::Vec3D> points = {
        {0.f, 0.f, 0.f},
        {0.f, 0.f, 1.f},
        {1.f, 0.f, 0.f},
        {1.f, 0.f, 1.f},
        {1.f, 1.f, 0.f},
        {1.f, 1.f, 1.f},
        {0.f, 1.f, 0.f},
        {0.f, 1.f, 1.f},
    };
    rr_stream.log("strips", rr::archetypes::LineStrips3D(points));
}
