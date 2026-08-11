#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace utils {

size_t number_of_digits(double n) {
  std::ostringstream strs;
  strs << n;
  return strs.str().size();
}

std::vector<std::string> generateColorGradient(size_t num_colors) {
  std::vector<std::string> colors;
  colors.reserve(num_colors);

  struct ColorPoint {
    int r, g, b;
    float position;
  };

  std::vector<ColorPoint> color_points = {
      {0, 128, 0, 0.0f},
      {0, 255, 0, 0.25f},
      {255, 255, 0, 0.5f},
      {255, 128, 0, 0.75f},
      {255, 0, 0, 1.0f}
  };

  for (size_t i = 0; i < num_colors; ++i) {
    float t = static_cast<float>(i) / (num_colors - 1);

    ColorPoint* lower = &color_points[0];
    ColorPoint* upper = &color_points[color_points.size() - 1];

    for (size_t j = 0; j < color_points.size() - 1; ++j) {
      if (t >= color_points[j].position && t <= color_points[j + 1].position) {
        lower = &color_points[j];
        upper = &color_points[j + 1];
        break;
      }
    }

    float local_t = (t - lower->position) / (upper->position - lower->position);
    int r = static_cast<int>(lower->r + local_t * (upper->r - lower->r));
    int g = static_cast<int>(lower->g + local_t * (upper->g - lower->g));
    int b = static_cast<int>(lower->b + local_t * (upper->b - lower->b));

    int color_code = 16 + (r / 51) * 36 + (g / 51) * 6 + (b / 51);
    colors.push_back("\033[48;5;" + std::to_string(color_code) + "m");
  }

  return colors;
}

std::vector<std::string> generateFixedGradient() {
  std::vector<std::string> colors;
  colors.reserve(32);

  struct ColorPoint {
    int r, g, b;
    float position;
  };

  std::vector<ColorPoint> color_points = {
      {0, 128, 0, 0.0f},
      {0, 255, 0, 0.25f},
      {255, 255, 0, 0.5f},
      {255, 128, 0, 0.75f},
      {255, 0, 0, 1.0f}
  };

  for (size_t i = 0; i < 32; ++i) {
    float t = static_cast<float>(i) / 31.0f;

    ColorPoint* lower = &color_points[0];
    ColorPoint* upper = &color_points[color_points.size() - 1];

    for (size_t j = 0; j < color_points.size() - 1; ++j) {
      if (t >= color_points[j].position && t <= color_points[j + 1].position) {
        lower = &color_points[j];
        upper = &color_points[j + 1];
        break;
      }
    }

    float local_t = (t - lower->position) / (upper->position - lower->position);
    int r = static_cast<int>(lower->r + local_t * (upper->r - lower->r));
    int g = static_cast<int>(lower->g + local_t * (upper->g - lower->g));
    int b = static_cast<int>(lower->b + local_t * (upper->b - lower->b));

    int color_code = 16 + (r / 51) * 36 + (g / 51) * 6 + (b / 51);
    colors.push_back("\033[48;5;" + std::to_string(color_code) + "m");
  }

  return colors;
}

template <size_t N, size_t M>
void printMatrix(const double matrix[N][M], size_t n, size_t m) {
  size_t max_len_per_column[M] = {0};

  for (size_t j = 0; j < m; ++j) {
    size_t max_len = 0;
    for (size_t i = 0; i < n; ++i) {
      if (const auto num_length = number_of_digits(matrix[i][j]);
          num_length > max_len) {
        max_len = num_length;
      }
    }
    max_len_per_column[j] = max_len;
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      std::cout << (j == 0 ? "\n| " : "") << std::setw(max_len_per_column[j])
                << matrix[i][j] << (j == m - 1 ? " |" : " ");
    }
  }
  std::cout << '\n';
}

void printMatrixHeatmap(const int* matrix, size_t height, size_t width,
                        size_t block_size) {
  const std::string reset_color = "\033[0m";
  auto colors = generateColorGradient(block_size * block_size);

  for (size_t i = 0; i < height; ++i) {
    std::cout << "\n| ";
    for (size_t j = 0; j < width; ++j) {
      int value = matrix[i * width + j];
      size_t color_idx = static_cast<size_t>(value) % colors.size();
      std::cout << colors[color_idx] << std::setw(2) << value << reset_color
                << " ";
    }
    std::cout << "|";
  }
  std::cout << "\n\nColor Scale: Green -> Yellow -> Orange -> Red\n";
  std::cout << "Colors scaled to block size: " << block_size << "x"
            << block_size << "\n";
}

void printMatrixHeatmap32(const int* matrix, size_t height, size_t width) {
  const std::string reset_color = "\033[0m";
  static auto colors = generateFixedGradient();

  for (size_t i = 0; i < height; ++i) {
    std::cout << "\n| ";
    for (size_t j = 0; j < width; ++j) {
      int value = matrix[i * width + j];
      size_t color_idx = static_cast<size_t>(value) % 32;
      std::cout << colors[color_idx] << std::setw(2) << value << reset_color
                << " ";
    }
    std::cout << "|";
  }
  std::cout << "\n\nColor Scale: Green -> Yellow -> Orange -> Red\n";
  std::cout << "Values range: 0-31\n";
}

void printMatrix(const int* matrix, size_t height, size_t width) {
  size_t max_len_per_column[width] = {0};

  for (size_t j = 0; j < width; ++j) {
    size_t max_len = 0;
    for (size_t i = 0; i < height; ++i) {
      if (const auto num_length = number_of_digits(matrix[i * width + j]);
          num_length > max_len) {
        max_len = num_length;
      }
    }
    max_len_per_column[j] = max_len;
  }

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      std::cout << (j == 0 ? "\n| " : "") << std::setw(max_len_per_column[j])
                << matrix[i * width + j] << (j == width - 1 ? " |" : " ");
    }
  }
  std::cout << '\n';
}

}