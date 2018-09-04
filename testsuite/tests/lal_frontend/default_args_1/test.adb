procedure Test is
   function F(X : Integer := 0) return Integer is
   begin
      return X;
   end F;

   R : Integer := F;
begin
   null;
end Ex1;
