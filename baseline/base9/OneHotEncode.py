from sklearn.preprocessing import OneHotEncoder
# import sklearn
class OneHotEncode:
    # features: 二维数组
    def __init__(self):
        self.onehot = OneHotEncoder(handle_unknown='ignore')

    def transform(self , features):
        return self.onehot.fit_transform(features).toarray()

if __name__ == '__main__':
    model = OneHotEncode()
    # print(sklearn.__version__)
    data = [['EI_EXPOSE_REP'], ['FE_FLOATING_POINT_EQUALITY'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['DMI_INVOKING_TOSTRING_ON_ARRAY'], ['UC_USELESS_CONDITION'], ['DCN_NULLPOINTER_EXCEPTION'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EQ_COMPARETO_USE_OBJECT_EQUALS'], ['RV_RETURN_VALUE_IGNORED_NO_SIDE_EFFECT'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['MS_EXPOSE_REP'], ['FE_FLOATING_POINT_EQUALITY'], ['DLS_DEAD_LOCAL_STORE'], ['BC_UNCONFIRMED_CAST_OF_RETURN_VALUE'], ['DLS_DEAD_LOCAL_STORE'], ['DMI_INVOKING_TOSTRING_ON_ARRAY'], ['DCN_NULLPOINTER_EXCEPTION'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EI_EXPOSE_REP'], ['RC_REF_COMPARISON_BAD_PRACTICE'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['RCN_REDUNDANT_NULLCHECK_OF_NONNULL_VALUE'], ['EI_EXPOSE_REP'], ['RC_REF_COMPARISON_BAD_PRACTICE'], ['EI_EXPOSE_REP'], ['ES_COMPARING_STRINGS_WITH_EQ'], ['BC_UNCONFIRMED_CAST'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['MS_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP'], ['RCN_REDUNDANT_NULLCHECK_WOULD_HAVE_BEEN_A_NPE'], ['ES_COMPARING_STRINGS_WITH_EQ'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EI_EXPOSE_REP'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['DLS_DEAD_LOCAL_STORE'], ['WMI_WRONG_MAP_ITERATOR'], ['EI_EXPOSE_REP'], ['ICAST_IDIV_CAST_TO_DOUBLE'], ['MS_EXPOSE_REP'], ['UL_UNRELEASED_LOCK_EXCEPTION_PATH'], ['BC_UNCONFIRMED_CAST_OF_RETURN_VALUE'], ['BC_UNCONFIRMED_CAST'], ['DCN_NULLPOINTER_EXCEPTION'], ['BC_UNCONFIRMED_CAST_OF_RETURN_VALUE'], ['DE_MIGHT_IGNORE'], ['EI_EXPOSE_REP2'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP'], ['NP_NULL_ON_SOME_PATH_FROM_RETURN_VALUE'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EI_EXPOSE_REP'], ['RV_RETURN_VALUE_IGNORED_NO_SIDE_EFFECT'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['ICAST_INTEGER_MULTIPLY_CAST_TO_LONG'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['CN_IMPLEMENTS_CLONE_BUT_NOT_CLONEABLE'], ['ICAST_IDIV_CAST_TO_DOUBLE'], ['WA_NOT_IN_LOOP'], ['RCN_REDUNDANT_NULLCHECK_OF_NONNULL_VALUE'], ['RCN_REDUNDANT_NULLCHECK_OF_NONNULL_VALUE'], ['EI_EXPOSE_REP2'], ['ICAST_IDIV_CAST_TO_DOUBLE'], ['EI_EXPOSE_REP'], ['RC_REF_COMPARISON_BAD_PRACTICE'], ['RCN_REDUNDANT_NULLCHECK_OF_NONNULL_VALUE'], ['DCN_NULLPOINTER_EXCEPTION'], ['ICAST_INTEGER_MULTIPLY_CAST_TO_LONG'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['ICAST_INTEGER_MULTIPLY_CAST_TO_LONG'], ['PZLA_PREFER_ZERO_LENGTH_ARRAYS'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['NP_NULL_PARAM_DEREF'], ['WA_NOT_IN_LOOP'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP2'], ['MS_EXPOSE_REP'], ['EI_EXPOSE_REP2'], ['MS_EXPOSE_REP'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EI_EXPOSE_STATIC_REP2'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['DCN_NULLPOINTER_EXCEPTION'], ['EI_EXPOSE_REP'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['BC_UNCONFIRMED_CAST'], ['EQ_COMPARETO_USE_OBJECT_EQUALS'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP2'], ['BC_UNCONFIRMED_CAST_OF_RETURN_VALUE'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EQ_COMPARETO_USE_OBJECT_EQUALS'], ['RCN_REDUNDANT_NULLCHECK_OF_NONNULL_VALUE'], ['NP_NULL_ON_SOME_PATH_FROM_RETURN_VALUE'], ['RV_RETURN_VALUE_IGNORED_NO_SIDE_EFFECT'], ['DM_NUMBER_CTOR'], ['DLS_DEAD_LOCAL_STORE'], ['DLS_DEAD_LOCAL_STORE'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['CN_IMPLEMENTS_CLONE_BUT_NOT_CLONEABLE'], ['MS_EXPOSE_REP'], ['FE_FLOATING_POINT_EQUALITY'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EQ_COMPARETO_USE_OBJECT_EQUALS'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['RC_REF_COMPARISON_BAD_PRACTICE'], ['BC_UNCONFIRMED_CAST'], ['NP_NULL_ON_SOME_PATH_FROM_RETURN_VALUE'], ['BC_UNCONFIRMED_CAST'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EQ_COMPARETO_USE_OBJECT_EQUALS'], ['EI_EXPOSE_REP2'], ['ICAST_INTEGER_MULTIPLY_CAST_TO_LONG'], ['DLS_DEAD_LOCAL_STORE'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['UC_USELESS_OBJECT'], ['RV_RETURN_VALUE_IGNORED_NO_SIDE_EFFECT'], ['BC_UNCONFIRMED_CAST'], ['DCN_NULLPOINTER_EXCEPTION'], ['GC_UNRELATED_TYPES'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['DE_MIGHT_IGNORE'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP'], ['RV_RETURN_VALUE_IGNORED_NO_SIDE_EFFECT'], ['RV_RETURN_VALUE_IGNORED_NO_SIDE_EFFECT'], ['FE_FLOATING_POINT_EQUALITY'], ['EI_EXPOSE_REP2'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['RCN_REDUNDANT_NULLCHECK_WOULD_HAVE_BEEN_A_NPE'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EI_EXPOSE_REP'], ['UC_USELESS_CONDITION'], ['EI_EXPOSE_REP'], ['UC_USELESS_CONDITION'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['ICAST_INTEGER_MULTIPLY_CAST_TO_LONG'], ['DLS_DEAD_LOCAL_STORE_OF_NULL'], ['EI_EXPOSE_REP'], ['DLS_DEAD_LOCAL_STORE'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP2'], ['EI_EXPOSE_REP'], ['DCN_NULLPOINTER_EXCEPTION'], ['EI_EXPOSE_REP'], ['DM_EXIT'], ['UC_USELESS_CONDITION'], ['BC_UNCONFIRMED_CAST'], ['PZLA_PREFER_ZERO_LENGTH_ARRAYS'], ['EI_EXPOSE_REP'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['DLS_DEAD_LOCAL_STORE'], ['EI_EXPOSE_REP'], ['DCN_NULLPOINTER_EXCEPTION'], ['ICAST_IDIV_CAST_TO_DOUBLE'], ['ICAST_IDIV_CAST_TO_DOUBLE'], ['EI_EXPOSE_REP'], ['RCN_REDUNDANT_NULLCHECK_WOULD_HAVE_BEEN_A_NPE'], ['EI_EXPOSE_REP'], ['ICAST_QUESTIONABLE_UNSIGNED_RIGHT_SHIFT'], ['VA_FORMAT_STRING_USES_NEWLINE'], ['EI_EXPOSE_REP'], ['NP_NULL_PARAM_DEREF'], ['EI_EXPOSE_REP'], ['WA_NOT_IN_LOOP'], ['BC_UNCONFIRMED_CAST'], ['RCN_REDUNDANT_NULLCHECK_OF_NONNULL_VALUE'], ['BC_UNCONFIRMED_CAST_OF_RETURN_VALUE'], ['EI_EXPOSE_REP'], ['DM_EXIT'], ['EI_EXPOSE_REP2'], ['FE_FLOATING_POINT_EQUALITY'], ['BC_UNCONFIRMED_CAST'], ['RV_RETURN_VALUE_IGNORED_BAD_PRACTICE'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['NP_NULL_ON_SOME_PATH_FROM_RETURN_VALUE'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['UL_UNRELEASED_LOCK_EXCEPTION_PATH'], ['EI_EXPOSE_REP2'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['BC_UNCONFIRMED_CAST_OF_RETURN_VALUE'], ['ICAST_INTEGER_MULTIPLY_CAST_TO_LONG'], ['UC_USELESS_CONDITION'], ['EI_EXPOSE_REP'], ['REC_CATCH_EXCEPTION'], ['RV_RETURN_VALUE_IGNORED_NO_SIDE_EFFECT'], ['EI_EXPOSE_REP'], ['DLS_DEAD_LOCAL_STORE'], ['ICAST_IDIV_CAST_TO_DOUBLE'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['DLS_DEAD_LOCAL_STORE_OF_NULL'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['UC_USELESS_CONDITION'], ['EI_EXPOSE_REP2'], ['ICAST_INTEGER_MULTIPLY_CAST_TO_LONG'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['BC_UNCONFIRMED_CAST'], ['RC_REF_COMPARISON_BAD_PRACTICE'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP2'], ['NP_NULL_PARAM_DEREF'], ['EI_EXPOSE_REP2'], ['GC_UNRELATED_TYPES'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['RV_RETURN_VALUE_IGNORED_NO_SIDE_EFFECT'], ['DLS_DEAD_LOCAL_STORE'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['BC_UNCONFIRMED_CAST'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['ICAST_INTEGER_MULTIPLY_CAST_TO_LONG'], ['DLS_DEAD_LOCAL_STORE'], ['EI_EXPOSE_REP'], ['RCN_REDUNDANT_NULLCHECK_OF_NONNULL_VALUE'], ['IP_PARAMETER_IS_DEAD_BUT_OVERWRITTEN'], ['NP_LOAD_OF_KNOWN_NULL_VALUE'], ['EI_EXPOSE_REP'], ['RCN_REDUNDANT_NULLCHECK_OF_NULL_VALUE'], ['ICAST_IDIV_CAST_TO_DOUBLE'], ['NP_NULL_PARAM_DEREF'], ['BC_UNCONFIRMED_CAST'], ['BC_UNCONFIRMED_CAST'], ['BC_UNCONFIRMED_CAST'], ['UG_SYNC_SET_UNSYNC_GET'], ['EI_EXPOSE_REP'], ['ICAST_QUESTIONABLE_UNSIGNED_RIGHT_SHIFT'], ['DLS_DEAD_LOCAL_STORE'], ['EI_EXPOSE_REP'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EI_EXPOSE_REP'], ['PZLA_PREFER_ZERO_LENGTH_ARRAYS'], ['BC_UNCONFIRMED_CAST'], ['PZLA_PREFER_ZERO_LENGTH_ARRAYS'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['DLS_DEAD_LOCAL_STORE'], ['EI_EXPOSE_REP'], ['EI_EXPOSE_REP'], ['NP_LOAD_OF_KNOWN_NULL_VALUE'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['UC_USELESS_CONDITION'], ['EI_EXPOSE_REP'], ['BC_UNCONFIRMED_CAST'], ['DCN_NULLPOINTER_EXCEPTION'], ['UWF_FIELD_NOT_INITIALIZED_IN_CONSTRUCTOR'], ['UC_USELESS_CONDITION'], ['EI_EXPOSE_REP'], ['DMI_INVOKING_TOSTRING_ON_ARRAY'], ['BC_UNCONFIRMED_CAST'], ['BC_UNCONFIRMED_CAST'], ['BC_UNCONFIRMED_CAST'], ['NP_LOAD_OF_KNOWN_NULL_VALUE'], ['UG_SYNC_SET_UNSYNC_GET'], ['VO_VOLATILE_INCREMENT'], ['BC_UNCONFIRMED_CAST']]

    # [['EI_EXPOSE_REP'], ['NONE' ], ['Female' ] , ['test' ] ,['Male' ]]
    res = model.transform(data)

    print(res)

    print(type(res))

    print(res.tolist())

    enc = OneHotEncoder()
    enc.fit([[0, 0, 3],
             [1, 1, 0],
             [0, 2, 1],
             [1, 0, 2]])


    ans = enc.transform([[0, 1, 3]]).toarray()
    print(ans)