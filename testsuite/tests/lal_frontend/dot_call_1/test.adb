procedure Test is
   package O is
      type A is tagged record
        E : Integer;
      end record;

      function Get_E(X : A) return Integer;
   end O;

   package body O is
      function Get_E(X : A) return Integer is
      begin
         return X.E;
      end Get_E;
   end O;

   P : O.A;
   R : Integer := P.Get_E;
begin
   null;
end Ex1;
