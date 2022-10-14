#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

using uint = unsigned int;

// some mathematical constants
constexpr double Pi() {
    return 3.14159265358979323846264338327950288419716939937510;
}  // 50 digits
constexpr double TwoPi() {
    return 2.0 * Pi();
}
constexpr double PiSq() {
    return Pi() * Pi();
}
constexpr double PiOver2() {
    return Pi() / 2.0;
}
constexpr double PiOver4() {
    return Pi() / 4.0;
}
constexpr double InvPi() {
    return 1.0 / Pi();
}
constexpr double RadToDeg() {
    return 180.0 / Pi();
}
constexpr double DegToRad() {
    return Pi() / 180.0;
}
constexpr double Sqrt2() {
    return 1.4142135623730950488016887242097;
}

// floating point comparison
template<
    typename Floating,
    typename = typename std::enable_if<std::is_floating_point<Floating>::value>>
bool equals(Floating A, Floating B) {
    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    Floating diff = fabs(A - B);
    if (diff <= 2 * std::numeric_limits<Floating>::epsilon())
        return true;

    A = fabs(A);
    B = fabs(B);
    Floating largest = (B > A) ? B : A;

    if (diff <= largest * std::numeric_limits<Floating>::epsilon())
        return true;
    return false;
}

// Some mathematical functions with unlimited range thanks to variadic templates
template<typename T>
inline T sq(T val) {
    return val * val;
}
template<typename T>
inline T square(T val) {
    return val * val;
}
// NOLINTNEXTLINE(runtime/int)
inline unsigned long long int operator"" _sq(unsigned long long int val) {
    return val * val;
}
inline long double operator"" _sq(long double val) {
    return val * val;
}

template<typename T>
inline T cube(T val) {
    return val * val * val;
}
// NOLINTNEXTLINE(runtime/int)
inline unsigned long long int operator"" _cube(unsigned long long int val) {
    return val * val * val;
}
inline long double operator"" _cube(long double val) {
    return val * val * val;
}

/// power with integer exponent
template<typename T>
inline T pow_n(T base, unsigned int exp) {
    if (exp == 0u) {
        return static_cast<T>(1);
    }
    T result = static_cast<T>(1);
    while (exp) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        base *= base;
    }
    return result;
}

/// Sum of squares, result given in double
template<typename T>
inline double sum2(T val) {
    return static_cast<double>(val) * static_cast<double>(val);
}
template<typename T, typename... Args>
inline double sum2(T val, Args... args) {
    return static_cast<double>(val) * static_cast<double>(val) + sum2(args...);
}

/// Norm 2, the square root of the sum of squares, result given in double
template<typename... Args>
inline double norm2(Args... args) {
    return std::sqrt(sum2(args...));
}

/// Difference of square
template<typename T, typename U>
inline double diff2(T a, U b) {
    return static_cast<double>(a) * static_cast<double>(a)
           - static_cast<double>(b) * static_cast<double>(b);
}

namespace detail_string
{
template<typename S, typename T, typename = void>
struct is_streamable: std::false_type {};

template<typename S, typename T>
struct is_streamable<
    S,
    T,
    std::void_t<decltype(std::declval<S&>() << std::declval<T>())>>: std::true_type {};

template<
    typename T,
    typename = typename std::enable_if<is_streamable<std::ostringstream, T>{}>::type>
inline void PushToStream(std::ostringstream& oss, T&& val) {
    oss << std::forward<T>(val);
}

template<
    typename T,
    typename... Args,
    typename = typename std::enable_if<is_streamable<std::ostringstream, T>{}>::type>
inline void PushToStream(std::ostringstream& oss, T&& val, Args&&... args) {
    oss << std::forward<T>(val);
    PushToStream(oss, std::forward<Args>(args)...);
}
}  // end namespace detail_string

//! Concatenation of y_ streamable objects into a string, returns a std::string
template<typename... Args>
inline std::string cat(Args&&... args) {
    std::ostringstream oss;
    detail_string::PushToStream(oss, std::forward<Args>(args)...);
    return oss.str();
}

//! An operator<< on all enum types, enabling easy printing to log or debug
template<typename Enum>
inline std::ostream& operator<<(
    typename std::enable_if<std::is_enum<Enum>::value, std::ostream>::type& stream,
    const Enum& e) {
    return stream << static_cast<typename std::underlying_type<Enum>::type>(e);
}

//! The local exception
class Error: public std::exception {
public:
    virtual ~Error() noexcept {}

    template<typename... Args>
    // NOLINTNEXTLINE
    Error(Args... args): std::exception() {
        message = cat(args...);
    }

    const std::string& swhat() const noexcept { return message; }
    const char* what() const noexcept override { return message.c_str(); }

private:
    std::string message;
};

// Some constants
constexpr uint HERO_MOV{800u};  // hero maximal movement per turn
constexpr uint MSTR_MOV{400u};  // monster maximal movement per turn
constexpr uint DMG{2u};  // damages per turn
constexpr uint R_DMG{800u};  // damage radius
constexpr uint R_BASE{300u};  // base radius
constexpr uint X_MIN{0u};
constexpr uint Y_MIN{0u};
constexpr uint X_MAX{17630u};
constexpr uint Y_MAX{9000u};

//! A 2 dimension point.
class Point {
    uint x_, y_;

public:
    Point(): x_(0u), y_(0u) {}
    Point(uint x, uint y): x_(x), y_(y) {
        if (x > X_MAX or y > Y_MAX) {
            throw Error(
                "Point out of bounds: ", x, " > ", X_MAX, " or ", y, " > ", Y_MAX);
        }
    }

    //! Create a Point from polar coordinates.
    static Point from_polar(double r, double theta) {
        return Point(r * cos(theta), r * sin(theta));
    }

    uint x() const { return x_; }
    uint y() const { return y_; }
    uint at(bool i) const { return i ? y_ : x_; }

    double r() const { return norm2(x_, y_); }  //! Modulus
    double theta() const { return atan2(x_, y_); }  //! Argument
    //! Distance to another point
    double dist_to(const Point& p) const { return norm2(x_ - p.x(), y_ - p.y()); }

    //! Advance the point from the given dist.
    void advance(uint dist) {
        const double t = theta();
        x_ += dist * cos(t);
        y_ += dist * sin(t);
        x_ = min(max(0u, x_), X_MAX);
        y_ = min(max(0u, y_), Y_MAX);
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "Point(" << p.x() << ", " << p.y() << ")";
        return os;
    }

    bool operator==(const Point& other) {
        return equals(x_, other.x()) and equals(y_, other.y());
    }
    Point operator+(const Point& other) {
        return Point(x_ + other.x(), y_ + other.y());
    }
    Point operator-(const Point& other) {
        return Point(x_ - other.x(), y_ - other.y());
    }
    Point operator*(uint d) { return Point(d * x_, d * y_); }
};

enum class Type
{
    kMonster,
    kHero,
    kOpponent
};

static const array<Point, 2> kBases{Point{X_MIN, Y_MIN}, Point{X_MAX, Y_MAX}};

//! An entity on the play field
class Entity {
    const uint id_;
    Point p_;
    const Type type_;

public:
    Entity(uint id, const Point& p, Type type): id_(id), p_(p), type_(type) {}
    virtual ~Entity() {}
    Entity(const Entity&) = default;
    constexpr Entity(Entity&&) = default;

    uint id() const { return id_; }
    Point p() const { return p_; }
    Type type() const { return type_; }

    double dist_to(const Point& p) const { return p_.dist_to(p); }

    void update(uint x, uint y) { p_ = Point(x, y); }

    friend std::ostream& operator<<(std::ostream& os, const Entity& e) {
        os << "Entity(" << e.id() << ", " << e.p().x() << ", " << e.p().y() << ")";
        return os;
    }
};

class Hero: public Entity {
public:
    Hero(uint id, const Point& p): Entity(id, p, Type::kHero) {}
    Hero(int id, int x, int y):
        Entity(
            static_cast<uint>(id),
            {static_cast<uint>(x), static_cast<uint>(y)},
            Type::kHero) {}

    friend std::ostream& operator<<(std::ostream& os, const Hero& h) {
        os << "Hero(" << h.id() << ", " << h.p().x() << ", " << h.p().y() << ")";
        return os;
    }
};

class Opponent: public Entity {
public:
    Opponent(uint id, const Point& p): Entity(id, p, Type::kOpponent) {}

    friend std::ostream& operator<<(std::ostream& os, const Opponent& o) {
        os << "Opponent(" << o.id() << ", " << o.p().x() << ", " << o.p().y() << ")";
        return os;
    }
};

enum class Threat
{
    kNone,
    kBase,
    kOpponent
};

class Monster: public Entity {
    uint health_;
    int vx_, vy_;
    bool has_target_;
    Threat threat_;

public:
    Monster(
        uint id,
        const Point& p,
        uint health,
        int vx,
        int vy,
        bool has_target,
        int threat):
        Entity(id, p, Type::kMonster),
        health_(health),
        vx_(vx),
        vy_(vy),
        has_target_(has_target),
        threat_(Threat(threat)) {}
    Monster(
        int id, int x, int y, int health, int vx, int vy, int has_target, int threat):
        Entity(
            static_cast<uint>(id),
            {static_cast<uint>(x), static_cast<uint>(y)},
            Type::kMonster),
        health_(static_cast<uint>(health)),
        vx_(vx),
        vy_(vy),
        has_target_(static_cast<bool>(has_target)),
        threat_(Threat(threat)) {}
    Monster(const Monster&) = default;
    constexpr Monster(Monster&&) = default;

    uint health() const { return health_; }
    int vx() const { return vx_; }
    int vy() const { return vy_; }
    bool has_target() const { return has_target_; }
    Threat threat() const { return threat_; }

    void update(
        uint x, uint y, uint health, int vx, int vy, bool has_target, int threat) {
        Entity::update(x, y);
        health_ = health;
        vx_ = vx;
        vy_ = vy;
        has_target_ = has_target;
        threat_ = Threat(threat);
    }

    friend std::ostream& operator<<(std::ostream& os, const Monster& m) {
        os << "Monster(" << m.id() << ", " << m.p().x() << ", " << m.p().y() << ", "
           << m.health() << ", " << m.vx() << ", " << m.vy() << ", "
           << (m.has_target() ? "targetting" : "wandering") << ", " << m.threat()
           << ")";
        return os;
    }
};

//! Return the distance between 2 Points.
double dist(const Point& a, const Point& b) {
    return norm2(a.x() - b.x(), a.y() - b.y());
}

//! Print to stderr to debug
template<typename... Args>
void debug(Args&&... args) {
    cerr << cat(args...) << endl;
}

Point standby(uint index, bool top_left) {
    const int sign = top_left ? 1 : -1;
    const Point home = top_left ? Point(X_MIN, Y_MIN) : Point(X_MAX, Y_MAX);
    constexpr uint a = static_cast<uint>(R_DMG * (1 + Sqrt2()));
    constexpr uint b = static_cast<uint>(R_DMG * (1. + 2. * cos(Pi() / 12.)));

    if (index == 0u) {
        return Point(home.x() + sign * b, home.y() + sign * b);
    } else if (index == 1u) {
        return Point(home.x() + sign * R_DMG, home.y() + sign * a);
    } else if (index == 2u) {
        return Point(home.x() + sign * a, home.y() + sign * R_DMG);
    } else {
        throw Error("Incorrect hero index (>3): ", index);
    }
}

struct {
    bool operator()(const Monster& a, const Monster& b) {
        auto d_a = a.dist_to(kBases[0]);
        auto d_b = b.dist_to(kBases[0]);
        return d_a > d_b;
    }
} cmpMonster;

int main() {
    int base_x;  // The corner of the map representing your base
    int base_y;
    cin >> base_x >> base_y;
    cin.ignore();
    int heroes_per_player;  // Always 3
    cin >> heroes_per_player;
    cin.ignore();

    bool top_left = base_x == 0 ? true : false;

    map<int, Hero> heroes;
    map<int, Monster> monsters;
    vector<reference_wrapper<Monster>> dangers;

    // game loop
    while (1) {
        monsters.clear();
        dangers.clear();
        for (int i = 0; i < 2; i++) {
            int health;  // Each player's base health
            int mana;  // Ignore in the first league; Spend ten mana to cast a spell
            cin >> health >> mana;
            cin.ignore();
        }
        int entity_count;  // Amount of heros and monsters you can see
        cin >> entity_count;
        cin.ignore();
        for (int i = 0; i < entity_count; i++) {
            int id;  // Unique identifier
            int type;  // 0=monster, 1=your hero, 2=opponent hero
            int x;  // Position of this entity
            int y;
            int shield_life;  // Ignore for this league; Count down until shield spell
                              // fades
            int is_controlled;  // Ignore for this league; Equals 1 when this entity is
                                // under a control spell
            int health;  // Remaining health of this monster
            int vx;  // Trajectory of this monster
            int vy;
            int has_target;  // 0=monster with no target yet, 1=monster targeting a base
            int threat;  // Given this monster's trajectory, is it a threat to
                         // 1=your base, 2=your opponent's base, 0=neither
            cin >> id >> type >> x >> y >> shield_life >> is_controlled >> health >> vx
                >> vy >> has_target >> threat;
            cin.ignore();
            if (type == 0) {  // monsters
                monsters.emplace(pair<int&, Monster>(
                    id, {id, x, y, health, vx, vy, has_target, threat}));
            } else if (type == 1) {  // heroes
                if (not heroes.count(id)) {
                    heroes.emplace(pair<int&, Hero>(id, {id, x, y}));
                } else {
                    heroes.at(id).update(x, y);
                }
            }
        }
        for (auto& [id, monster] : monsters) {
            if (monster.threat() == Threat::kBase) {
                dangers.emplace_back(ref(monster));
            }
        }
        sort(dangers.begin(), dangers.end(), cmpMonster);
        for (int i = 0; i < heroes_per_player; i++) {
            // Write an action using cout. DON'T FORGET THE "<< endl"
            // To debug: cerr << "Debug messages..." << endl;
            Point dest;
            if (dangers.empty()) {
                dest = standby(i, top_left);
            } else {
                dest = dangers.back().get().p();
                dangers.pop_back();
            }

            // In the first league: MOVE <x> <y> | WAIT; In later leagues: | SPELL
            // <spellParams>;
            cout << "MOVE " << dest.x() << " " << dest.y() << endl;
        }
    }
}
