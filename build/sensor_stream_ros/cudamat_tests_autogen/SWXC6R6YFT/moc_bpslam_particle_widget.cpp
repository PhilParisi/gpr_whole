/****************************************************************************
** Meta object code from reading C++ file 'bpslam_particle_widget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "include/sensor_stream_ros/ui/bpslam_particle_widget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'bpslam_particle_widget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_BpslamParticleWidget_t {
    QByteArrayData data[9];
    char stringdata0[185];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_BpslamParticleWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_BpslamParticleWidget_t qt_meta_stringdata_BpslamParticleWidget = {
    {
QT_MOC_LITERAL(0, 0, 20), // "BpslamParticleWidget"
QT_MOC_LITERAL(1, 21, 20), // "pubHypothesisClicked"
QT_MOC_LITERAL(2, 42, 0), // ""
QT_MOC_LITERAL(3, 43, 25), // "on_hypothesis_btn_clicked"
QT_MOC_LITERAL(4, 69, 18), // "on_gpr_btn_clicked"
QT_MOC_LITERAL(5, 88, 27), // "on_gpr_training_btn_clicked"
QT_MOC_LITERAL(6, 116, 24), // "on_recompute_gpr_clicked"
QT_MOC_LITERAL(7, 141, 19), // "on_mean_btn_clicked"
QT_MOC_LITERAL(8, 161, 23) // "on_save_map_btn_clicked"

    },
    "BpslamParticleWidget\0pubHypothesisClicked\0"
    "\0on_hypothesis_btn_clicked\0"
    "on_gpr_btn_clicked\0on_gpr_training_btn_clicked\0"
    "on_recompute_gpr_clicked\0on_mean_btn_clicked\0"
    "on_save_map_btn_clicked"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_BpslamParticleWidget[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   49,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    0,   50,    2, 0x08 /* Private */,
       4,    0,   51,    2, 0x08 /* Private */,
       5,    0,   52,    2, 0x08 /* Private */,
       6,    0,   53,    2, 0x08 /* Private */,
       7,    0,   54,    2, 0x08 /* Private */,
       8,    0,   55,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void BpslamParticleWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<BpslamParticleWidget *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->pubHypothesisClicked(); break;
        case 1: _t->on_hypothesis_btn_clicked(); break;
        case 2: _t->on_gpr_btn_clicked(); break;
        case 3: _t->on_gpr_training_btn_clicked(); break;
        case 4: _t->on_recompute_gpr_clicked(); break;
        case 5: _t->on_mean_btn_clicked(); break;
        case 6: _t->on_save_map_btn_clicked(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (BpslamParticleWidget::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&BpslamParticleWidget::pubHypothesisClicked)) {
                *result = 0;
                return;
            }
        }
    }
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject BpslamParticleWidget::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_BpslamParticleWidget.data,
    qt_meta_data_BpslamParticleWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *BpslamParticleWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *BpslamParticleWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_BpslamParticleWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int BpslamParticleWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void BpslamParticleWidget::pubHypothesisClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
