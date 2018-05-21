procedure Ex1 is
   package O is
      type A is tagged record
        E : Integer;
      end record;

      procedure Set_E(X : out A);
   end O;

   package body O is
      procedure Set_E(X : out A) is
      begin
         X.E := 3;
      end Set_E;
   end O;

   P : O.A;
begin
   P.Set_E;
end Ex1;
