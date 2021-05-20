#include "ApplicationConfiguration.hpp"

std::ostream& operator<<(std::ostream& os, 
    const ApplicationConfiguration& configuration)
{
    os << "Vocabulary:          " << configuration.VocabularyPath << "\n";
    os << "SVO:                 " << configuration.SVOPath << "\n";
    os << "Configuration:       " << configuration.ConfigurationPath << "\n";

    os << "Map:                 " << configuration.MapPath << "\n";
    os << "Frame trajectory:    " << configuration.FrameTrajectoryPath << "\n";
    os << "Keyframe trajectory: " << configuration.KeyframeTrajectoryPath 
        << "\n";
    os << "Track times:         " << configuration.TrackTimesPath << "\n";
    os << "Process times:       " << configuration.ProcessTimesPath << "\n";

    os << "Frame skip:          " << configuration.FrameSkip << "\n";
    os << "Frame start:         " << configuration.FrameStart << "\n";
    os << "Frame stop:          " << configuration.FrameStop<< "\n";

    os << "Filter diameter:     " << configuration.FilterDiameter << "\n";
    os << "Filter color:        " << configuration.FilterColor << "\n";
    os << "Filter space:        " << configuration.FilterSpace << "\n";

    os << "Realtime:            " << configuration.Realtime << "\n";
    os << "Terminate:           " << configuration.Terminate << "\n";
    os << "Debug:               " << configuration.Debug << "\n";
    os << "Evaluate:            " << configuration.Evaluate << "\n";

    os << "Preprocessing:       " << configuration.Preprocessing << "\n";

    return os;
}
